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
import tempfile
import os

from app.config import config
try:
    from app.models.asr_model import NeMoASRModel
    asr_model = NeMoASRModel("nvidia/parakeet-tdt-0.6b-v2")
except ImportError:
    asr_model = None

try:
    from app.models.tts_model import KokoroTTSModel
    tts_model = KokoroTTSModel()
except ImportError:
    tts_model = None

# Global queues for Dashboard-Pi interaction
pending_commands = []
pending_responses = []
from app.schemas import (
    HealthResponse,
    ModelInfo,
    ModelControlRequest,
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
            self._default_completion = FallbackModel(config.DEFAULT_COMPLETION_MODEL, "vision_language")
            self.loaded_models[config.DEFAULT_COMPLETION_MODEL] = self._default_completion

        def initialize_defaults(self):
            logger.info("Initialized fallback default models.")

        def get_model(self, model_name: Optional[str] = None) -> Any:
            if not model_name:
                return self._default_completion
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

    # 3. Initialize ASR Model
    if asr_model:
        try:
            asr_model.load()
        except Exception as e:
            logger.error(f"Failed to load ASR model during startup: {e}")
            
    # 4. Initialize TTS Model
    if tts_model:
        try:
            tts_model.load()
        except Exception as e:
            logger.error(f"Failed to load TTS model during startup: {e}")

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
        models_dict = model_registry.list_models()
        if isinstance(models_dict, dict):
            loaded_names = [name for name, info in models_dict.items() if isinstance(info, dict) and info.get("is_loaded")]
        elif isinstance(models_dict, list):
            loaded_names = [str(m) for m in models_dict]

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


@app.post("/api/v1/models/load", tags=["Models"])
async def api_load_model(req: ModelControlRequest):
    """
    Manually load a registered model into VRAM.
    """
    try:
        model_registry.load_model(req.model_name)
        return {"status": "success", "message": f"Successfully loaded {req.model_name} into VRAM."}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found in registry.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/unload", tags=["Models"])
async def api_unload_model(req: ModelControlRequest):
    """
    Manually unload a registered model from VRAM to free GPU memory.
    """
    try:
        model_registry.unload_model(req.model_name)
        return {"status": "success", "message": f"Successfully unloaded {req.model_name} from VRAM."}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found in registry.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/api/v1/communicate/transcribe", tags=["Communicate"])
async def transcribe_audio(audio_file: UploadFile = File(...)):
    if not asr_model or not asr_model.is_loaded:
        raise HTTPException(status_code=500, detail="ASR Model is not loaded or unavailable.")
    
    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise HTTPException(status_code=500, detail="imageio_ffmpeg not installed.")
        
    try:
        # Save the uploaded webm/ogg file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
            content = await audio_file.read()
            tmp_in.write(content)
            tmp_in_path = tmp_in.name
            
        # Convert to 16kHz mono WAV using raw ffmpeg executable
        tmp_out_path = tmp_in_path + ".wav"
        
        try:
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_out_path],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed. STDERR: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e.stderr}")
        
        # Transcribe using NeMo
        transcription = asr_model.transcribe(tmp_out_path)
        
        if not transcription:
            logger.warning("Transcription completed successfully but returned an empty string (likely silence).")
        else:
            logger.info(f"Transcription successful: {transcription}")
            pending_commands.append(transcription)
        
        # Cleanup
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)
        
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/commands/pop", tags=["Communicate"])
async def pop_command():
    if pending_commands:
        return {"command": pending_commands.pop(0)}
    return {"command": None}

@app.get("/api/v1/responses/pop", tags=["Communicate"])
async def pop_response():
    if pending_responses:
        return pending_responses.pop(0)
    return {"response": None}

import json

@app.post("/api/v1/communicate/action", tags=["Communicate"])
async def process_action(
    command: str = Form(...),
    image_file: UploadFile = File(...)
):
    """
    Called by the Pi script. It sends the image and the text command.
    Generates a structured JSON response using Gemma, then generates TTS audio using Kokoro.
    Pushes the response to the pending_responses queue for the Dashboard to read.
    """
    try:
        # Load image
        img_bytes = await image_file.read()
        loop = asyncio.get_running_loop()
        image = await loop.run_in_executor(None, decode_image, img_bytes)
        
        # Get Gemma model
        model = model_registry.get_model()
        if not model:
            raise HTTPException(status_code=500, detail="VLM Model not found.")
            
        system_prompt = (
            "You are a model hosted on a server, but takes images from a camera. "
            "You will get a text input, and as a response, you will reply text in the format of a json. "
            "In the json, there should be a speech object and an action object. "
            "In the speech object, return a SHORT response to the user, as well as finishing with: \"I am doing (such action) now\". "
            "In the action object, there should be two parameters: direction and magnitude. "
            "For direction, you can do right, left, front, or back. "
            "For magnitude, front and back units are in feet, and left and right units are in angle degrees."
        )
        
        # Call VLM
        result = await loop.run_in_executor(
            None,
            lambda: model.generate_completion(
                image=image,
                prompt=command,
                temperature=0.7,
                max_tokens=512,
                msgs=[{"role": "system", "content": system_prompt}]
            )
        )
        
        raw_text = result["choices"][0]["message"]["content"]
        
        # Parse JSON
        try:
            # Strip markdown formatting if any
            clean_text = raw_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            parsed_json = json.loads(clean_text)
            speech_text = parsed_json.get("Speech", "I am doing the action now.")
        except Exception as e:
            logger.error(f"Failed to parse JSON from Gemma: {e}. Raw text: {raw_text}")
            parsed_json = {"Speech": "I could not understand that.", "Action": {"Direction": "None", "Magnitude": "0"}}
            speech_text = parsed_json["Speech"]
            
        # Synthesize audio
        audio_base64 = ""
        if tts_model and tts_model.is_loaded:
            try:
                audio_base64 = await loop.run_in_executor(None, tts_model.synthesize_base64, speech_text)
            except Exception as e:
                logger.error(f"TTS failed: {e}")
                
        response_payload = {
            "transcription": command,
            "speech": speech_text,
            "action": parsed_json.get("Action", {}),
            "audio_base64": audio_base64
        }
        
        pending_responses.append(response_payload)
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Action processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server on {config.HOST}:{config.PORT}...")
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, reload=False)
