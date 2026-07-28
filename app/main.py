import asyncio
import base64
import io
import json
import logging
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import config
from app.schemas import (
    ActionInferenceRequest,
    ActionInferenceResponse,
    BoundingBox,
    DetectionInferenceResponse,
    HealthResponse,
    ModelInfo,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
)
from app.tunnel import CloudflareTunnelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

# Attempt to import model_registry; provide graceful fallback if models package is not yet created
try:
    from app.models.registry import model_registry
except ImportError:
    logger.warning("Could not import 'app.models.registry'. Using FallbackModelRegistry.")

    class FallbackModel:
        def __init__(self, name: str, task: str):
            self.name = name
            self.task = task

        def predict_action(self, image: Image.Image, candidate_actions: List[str]) -> Dict[str, Any]:
            time.sleep(0.01)  # Simulate inference latency
            top_action = candidate_actions[0] if candidate_actions else "stop"
            scores = {act: 1.0 / len(candidate_actions) for act in candidate_actions} if candidate_actions else {"stop": 1.0}
            if top_action in scores:
                scores[top_action] = 0.85
            return {
                "action": top_action,
                "confidence": scores.get(top_action, 1.0),
                "all_scores": scores,
                "model_used": self.name,
            }

        def detect(self, image: Image.Image) -> Dict[str, Any]:
            time.sleep(0.01)
            return {
                "bounding_boxes": [
                    {"xmin": 100.0, "ymin": 150.0, "xmax": 300.0, "ymax": 400.0, "confidence": 0.92, "class_name": "obstacle", "class_id": 0}
                ],
                "model_used": self.name,
            }

        def generate_completion(
            self,
            image: Optional[Image.Image] = None,
            prompt: str = "Describe the visual contents of this image.",
            temperature: float = 0.7,
            max_tokens: int = 512,
            msgs: Optional[List[Dict[str, Any]]] = None,
        ) -> Dict[str, Any]:
            time.sleep(0.01)
            res_str = f"[Fallback Mode] Simulated vision completion for prompt: '{prompt}'."
            return {
                "id": f"cmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.name,
                "choices": [
                    {
                        "index": 0,
                        "text": res_str,
                        "message": {"role": "assistant", "content": res_str},
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": max(1, len(prompt.split())),
                    "completion_tokens": max(1, len(res_str.split())),
                    "total_tokens": max(1, len(prompt.split())) + max(1, len(res_str.split()))
                },
                "inference_time_ms": 10.0,
                "raw_response": res_str
            }

    class FallbackModelRegistry:
        def __init__(self):
            self.loaded_models: Dict[str, Any] = {}
            self._default_action = FallbackModel(config.DEFAULT_ACTION_MODEL, "action_prediction")
            self._default_detect = FallbackModel(config.DEFAULT_DETECTION_MODEL, "object_detection")
            self.loaded_models[config.DEFAULT_ACTION_MODEL] = self._default_action
            self.loaded_models[config.DEFAULT_DETECTION_MODEL] = self._default_detect

        def initialize_defaults(self):
            logger.info("Initialized fallback default models.")

        def get_model(self, model_name: Optional[str] = None) -> Any:
            if not model_name:
                return self._default_action
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]
            # Auto return a fallback instance if requested
            model = FallbackModel(model_name, "general")
            self.loaded_models[model_name] = model
            return model

        def list_models(self) -> List[Dict[str, Any]]:
            return [
                {"name": name, "task_type": getattr(m, "task", "general"), "is_loaded": True, "device": "cuda" if torch.cuda.is_available() and config.USE_FP16 else "cpu"}
                for name, m in self.loaded_models.items()
            ]

        def cleanup(self):
            self.loaded_models.clear()

    model_registry = FallbackModelRegistry()


def decode_image(img_bytes: bytes) -> Image.Image:
    """
    Decodes raw image bytes or JPEG frames into an RGB PIL Image.
    """
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to decode image bytes: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image payload. Ensure valid JPEG or PNG bytes.")


def fetch_image_by_url(url: str) -> bytes:
    """
    Downloads raw image bytes from an HTTP or HTTPS link.
    """
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.read()
    except Exception as e:
        logger.error(f"Failed to fetch image from link '{url}': {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not download image from link: {url}")



def parse_candidate_actions(raw_actions: Any) -> List[str]:
    """
    Parses candidate actions from either a JSON string, comma-separated string, or list.
    """
    if isinstance(raw_actions, list):
        return [str(a).strip() for a in raw_actions if str(a).strip()]
    if isinstance(raw_actions, str):
        raw_str = raw_actions.strip()
        if raw_str.startswith("[") and raw_str.endswith("]"):
            try:
                parsed = json.loads(raw_str)
                if isinstance(parsed, list):
                    return [str(a).strip() for a in parsed if str(a).strip()]
            except json.JSONDecodeError:
                pass
        # Fallback to comma-separated splitting
        return [a.strip() for a in raw_str.split(",") if a.strip()]
    return ["move forward", "turn left", "turn right", "stop"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager.
    Handles startup (Cloudflare tunnel launch and model loading) and graceful shutdown.
    """
    logger.info("=== Starting Real-Time AI CV Robot PC-Side Framework ===")
    
    # 1. Initialize and start Cloudflare Tunnel
    if config.ENABLE_CLOUDFLARE or config.CLOUDFLARE_TUNNEL_TOKEN:
        app.state.tunnel = CloudflareTunnelManager(
            port=config.PORT,
            token=config.CLOUDFLARE_TUNNEL_TOKEN,
            static_domain=config.STATIC_DOMAIN,
        )
        try:
            tunnel_url = app.state.tunnel.start()
            logger.info(f"Cloudflare Tunnel Manager initialized. Public URL: {tunnel_url}")
        except Exception as e:
            logger.error(f"Could not start Cloudflare tunnel: {e}")
    else:
        app.state.tunnel = None
        logger.info(f"Cloudflare Tunnel is disabled. API is running locally on http://{config.HOST}:{config.PORT}")

    # 2. Initialize Model Registry and preload default models
    try:
        model_registry.initialize_defaults()
        logger.info("AI Model Registry initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing model registry: {e}")

    yield

    # Shutdown sequence
    logger.info("=== Shutting down PC-Side Framework ===")
    if hasattr(app.state, "tunnel") and app.state.tunnel:
        app.state.tunnel.stop()

    if hasattr(model_registry, "cleanup"):
        model_registry.cleanup()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("PyTorch CUDA VRAM cache cleared.")


app = FastAPI(
    title="Real-Time AI CV Robot Framework API",
    description="PC-side core backend and networking layer for low-latency AI robot navigation and vision.",
    version="1.2.3",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """
    Returns system diagnostics, GPU hardware status, VRAM allocation, and active model lists.
    """
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    vram_used = 0.0
    if gpu_available:
        try:
            vram_used = float(torch.cuda.memory_allocated(0) / (1024 ** 2))
        except Exception:
            vram_used = 0.0

    # Retrieve loaded model names
    loaded_names = []
    if hasattr(model_registry, "loaded_models"):
        if isinstance(model_registry.loaded_models, dict):
            loaded_names = list(model_registry.loaded_models.keys())
        elif isinstance(model_registry.loaded_models, list):
            loaded_names = [str(m) for m in model_registry.loaded_models]
    elif hasattr(model_registry, "list_models"):
        loaded_names = [m["name"] for m in model_registry.list_models() if m.get("is_loaded")]

    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        loaded_models=loaded_names,
        vram_used_mb=round(vram_used, 2),
    )


@app.get("/api/v1/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models():
    """
    Lists all available and loaded AI vision models in the registry.
    """
    if hasattr(model_registry, "list_models"):
        raw_list = model_registry.list_models()
        return [ModelInfo(**m) if isinstance(m, dict) else m for m in raw_list]
    
    # Fallback inspection
    models = []
    loaded = getattr(model_registry, "loaded_models", {})
    if isinstance(loaded, dict):
        for name in loaded.keys():
            models.append(ModelInfo(
                name=name,
                task_type="action_prediction" if "clip" in name.lower() or "action" in name.lower() else "object_detection",
                is_loaded=True,
                device="cuda" if torch.cuda.is_available() and config.USE_FP16 else "cpu",
            ))
    return models


@app.post("/api/v1/infer/action", response_model=ActionInferenceResponse, tags=["Inference"])
async def infer_action(request: Request):
    """
    Action prediction endpoint for robot navigation.
    Supports EITHER Multipart Form Data (file upload + candidate_actions form field)
    OR JSON body with base64 encoded image string.
    """
    start_time = time.perf_counter()
    content_type = request.headers.get("content-type", "").lower()
    
    img_bytes: Optional[bytes] = None
    candidate_actions: List[str] = []
    model_name: Optional[str] = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            req_data = ActionInferenceRequest(**body)
            candidate_actions = req_data.candidate_actions
            model_name = req_data.model_name
            if not req_data.image_base64:
                raise HTTPException(status_code=400, detail="image_base64 is required in JSON mode.")
            img_bytes = base64.b64decode(req_data.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON action request: {e}")
    else:
        # Assume Multipart Form Data or URL Encoded Form
        try:
            form = await request.form()
            file_field = form.get("file")
            actions_field = form.get("candidate_actions")
            model_field = form.get("model_name")

            if not file_field or not hasattr(file_field, "read"):
                raise HTTPException(status_code=400, detail="Missing image 'file' in multipart upload.")
            
            img_bytes = await file_field.read()
            candidate_actions = parse_candidate_actions(actions_field)
            model_name = str(model_field) if model_field else None
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse form upload: {e}")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="No image bytes provided.")

    # Execute decoding and inference in thread pool to prevent blocking asyncio loop
    loop = asyncio.get_running_loop()
    image = await loop.run_in_executor(None, decode_image, img_bytes)
    
    model = model_registry.get_model(model_name or config.DEFAULT_ACTION_MODEL)
    if not model or not hasattr(model, "predict_action"):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or does not support predict_action.")

    result = await loop.run_in_executor(None, model.predict_action, image, candidate_actions)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return ActionInferenceResponse(
        action=result.get("action", "stop"),
        confidence=float(result.get("confidence", 0.0)),
        all_scores=result.get("all_scores", {}),
        model_used=result.get("model_used", str(model_name or config.DEFAULT_ACTION_MODEL)),
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/v1/infer/detect", response_model=DetectionInferenceResponse, tags=["Inference"])
async def infer_detect(request: Request):
    """
    Object detection endpoint (YOLOv8). Returns bounding boxes, class labels, and confidence scores.
    Supports EITHER Multipart Form Data (file upload or image_url) OR JSON body (image_base64 or image_url).
    """
    start_time = time.perf_counter()
    content_type = request.headers.get("content-type", "").lower()
    
    img_bytes: Optional[bytes] = None
    model_name: Optional[str] = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            model_name = body.get("model_name") or body.get("model")
            if body.get("image_base64"):
                img_bytes = base64.b64decode(body["image_base64"])
            elif body.get("image_url"):
                img_bytes = fetch_image_by_url(body["image_url"])
            else:
                raise HTTPException(status_code=400, detail="Either image_base64 or image_url is required in JSON mode.")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON detect request: {e}")
    else:
        try:
            form = await request.form()
            file_field = form.get("file")
            url_field = form.get("image_url")
            model_field = form.get("model_name") or form.get("model")

            if file_field and hasattr(file_field, "read"):
                img_bytes = await file_field.read()
            elif url_field:
                img_bytes = fetch_image_by_url(str(url_field))
            model_name = str(model_field) if model_field else None
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse form upload: {e}")

    if not img_bytes:
        raise HTTPException(status_code=400, detail="No image bytes provided.")

    loop = asyncio.get_running_loop()
    image = await loop.run_in_executor(None, decode_image, img_bytes)
    
    target_model_name = model_name or config.DEFAULT_DETECTION_MODEL
    model = model_registry.get_model(target_model_name)
    if not model or not hasattr(model, "detect"):
        raise HTTPException(status_code=404, detail=f"Detection model '{target_model_name}' not found.")

    result = await loop.run_in_executor(None, model.detect, image)
    
    raw_boxes = result.get("bounding_boxes", [])
    formatted_boxes = []
    for b in raw_boxes:
        if isinstance(b, dict):
            formatted_boxes.append(BoundingBox(**b))
        elif isinstance(b, BoundingBox):
            formatted_boxes.append(b)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return DetectionInferenceResponse(
        bounding_boxes=formatted_boxes,
        model_used=result.get("model_used", target_model_name),
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.websocket("/api/v1/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-Time Video & Action Inference WebSocket Endpoint (< 50ms target).
    
    Accepts continuous video frames from the Raspberry Pi client in either:
    1. Binary mode: Raw JPEG/PNG frame bytes. Uses default candidate actions or pre-negotiated actions.
    2. Text/JSON mode: JSON packet containing `image_base64`, `candidate_actions`, and optional `model_name`.
    
    Returns real-time JSON decisions containing the predicted action and confidence scores.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected from {websocket.client}")
    
    # Default state for binary streaming
    active_model_name = config.DEFAULT_ACTION_MODEL
    default_candidate_actions = ["move forward", "turn left", "turn right", "stop"]
    loop = asyncio.get_running_loop()

    try:
        while True:
            message = await websocket.receive()
            start_time = time.perf_counter()

            img_bytes: Optional[bytes] = None
            candidate_actions = default_candidate_actions
            model_name = active_model_name

            if "bytes" in message and message["bytes"]:
                img_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if "image_base64" in data and data["image_base64"]:
                        img_bytes = base64.b64decode(data["image_base64"])
                    if "candidate_actions" in data and data["candidate_actions"]:
                        candidate_actions = parse_candidate_actions(data["candidate_actions"])
                        default_candidate_actions = candidate_actions
                    if "model_name" in data and data["model_name"]:
                        model_name = str(data["model_name"])
                        active_model_name = model_name
                except Exception as e:
                    await websocket.send_json({"error": f"Malformed JSON packet: {e}"})
                    continue

            if not img_bytes:
                await websocket.send_json({"error": "Empty frame or missing image data."})
                continue

            try:
                image = await loop.run_in_executor(None, decode_image, img_bytes)
                model = model_registry.get_model(model_name)
                if not model or not hasattr(model, "predict_action"):
                    await websocket.send_json({"error": f"Model '{model_name}' not ready."})
                    continue

                result = await loop.run_in_executor(None, model.predict_action, image, candidate_actions)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                response_packet = {
                    "action": result.get("action", "stop"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "all_scores": result.get("all_scores", {}),
                    "model_used": result.get("model_used", model_name),
                    "inference_time_ms": round(elapsed_ms, 2),
                    "timestamp": time.time(),
                }
                await websocket.send_json(response_packet)

            except Exception as inf_err:
                logger.error(f"WebSocket inference error: {inf_err}")
                await websocket.send_json({"error": str(inf_err)})

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected cleanly: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket unexpected error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/v1/completions", response_model=CompletionResponse, tags=["Completions"])
@app.post("/completions", response_model=CompletionResponse, tags=["Completions"])
@app.post("/v1/chat/completions", response_model=CompletionResponse, tags=["Completions"])
async def infer_completion(request: Request):
    """
    Vision-Language AI completion endpoint (OpenAI / v1/chat/completions compatible).
    Supports EITHER JSON body (messages array or prompt + image_base64)
    OR Multipart Form Data (file upload + prompt / model_name form fields).
    """
    start_time = time.perf_counter()
    content_type = request.headers.get("content-type", "").lower()

    img_bytes: Optional[bytes] = None
    prompt: str = "Describe the visual contents of this image."
    model_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    msgs: Optional[List[Dict[str, Any]]] = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            req_data = CompletionRequest(**body)
            model_name = req_data.model
            temperature = req_data.temperature
            max_tokens = req_data.max_tokens

            if req_data.prompt:
                prompt = req_data.prompt

            # Check for image_base64 or image_url field
            if req_data.image_base64:
                img_bytes = base64.b64decode(req_data.image_base64)
            elif req_data.image_url:
                img_bytes = fetch_image_by_url(req_data.image_url)

            # Check OpenAI-style messages array for prompt and image blocks
            if req_data.messages:
                msgs = [m.model_dump() if hasattr(m, "model_dump") else (m.dict() if hasattr(m, "dict") else m) for m in req_data.messages]
                # Try to extract text and image from messages if not already set
                for m in req_data.messages:
                    if m.role == "user" or not req_data.prompt:
                        content = m.content
                        if isinstance(content, str):
                            prompt = content
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    b_type = block.get("type", "")
                                    if b_type == "text" and "text" in block:
                                        prompt = block["text"]
                                    elif b_type in ("image_url", "image") and not img_bytes:
                                        img_val = block.get("image_url", block.get("image", {}))
                                        url_str = img_val.get("url", "") if isinstance(img_val, dict) else str(img_val)
                                        if url_str.startswith("data:image"):
                                            base64_part = url_str.split(",", 1)[-1]
                                            img_bytes = base64.b64decode(base64_part)
                                        elif url_str.startswith("http://") or url_str.startswith("https://"):
                                            img_bytes = fetch_image_by_url(url_str)
                                        else:
                                            img_bytes = base64.b64decode(url_str)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON completion request: {e}")
    else:
        # Multipart Form Data or URL Encoded Form
        try:
            form = await request.form()
            file_field = form.get("file")
            url_field = form.get("image_url")
            prompt_field = form.get("prompt")
            model_field = form.get("model") or form.get("model_name")
            temp_field = form.get("temperature")
            max_field = form.get("max_tokens") or form.get("max_new_tokens")

            if file_field and hasattr(file_field, "read"):
                img_bytes = await file_field.read()
            elif url_field:
                img_bytes = fetch_image_by_url(str(url_field))
            if prompt_field:
                prompt = str(prompt_field)
            if model_field:
                model_name = str(model_field)
            if temp_field:
                try:
                    temperature = float(temp_field)
                except ValueError:
                    pass
            if max_field:
                try:
                    max_tokens = int(max_field)
                except ValueError:
                    pass
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse form upload: {e}")

    loop = asyncio.get_running_loop()
    image: Optional[Image.Image] = None
    if img_bytes:
        image = await loop.run_in_executor(None, decode_image, img_bytes)

    target_model_name = model_name or "openbmb/MiniCPM-V"
    model = model_registry.get_model(target_model_name)
    if not model or not hasattr(model, "generate_completion"):
        # Fall back to default model in registry if requested model not found
        model = model_registry.get_model()
        if not model or not hasattr(model, "generate_completion"):
            raise HTTPException(status_code=404, detail=f"Model '{target_model_name}' not found or does not support completions.")

    result = await loop.run_in_executor(
        None,
        lambda: model.generate_completion(
            image=image,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            msgs=msgs
        )
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    result["inference_time_ms"] = round(elapsed_ms, 2)
    return CompletionResponse(**result)
