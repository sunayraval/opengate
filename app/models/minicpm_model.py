"""
MiniCPM-V Model Implementation for Multimodal Zero-Shot Action Classification & Vision Reasoning.
Provides high-performance inference with automatic BF16/FP16 precision optimization on CUDA devices.
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
from app.models.base import BaseVisionModel

logger = logging.getLogger(__name__)


class MiniCPMVModel(BaseVisionModel):
    """
    MiniCPM-V Vision-Language Model implementation for autonomous robot navigation and action classification.
    Leverages openbmb/MiniCPM-V with automatic BF16/FP16 quantization for efficient deployment.
    """

    def __init__(
        self,
        name: str = "openbmb/MiniCPM-V",
        model_path: str = "openbmb/MiniCPM-V",
        use_fp16: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize MiniCPM-V multimodal vision-language model.
        
        Args:
            name: Registry identifier for this model instance.
            model_path: Hugging Face repository ID or local path (e.g., 'openbmb/MiniCPM-V').
            use_fp16: Whether to use half-precision (BF16 or FP16) on GPU.
            device: Target computation device ('cuda' or 'cpu').
        """
        super().__init__(name=name, model_type="action", device=device)
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.model = None
        self.tokenizer = None

    def _get_target_dtype(self) -> torch.dtype:
        """Determine optimal PyTorch data type based on hardware capabilities."""
        if self.device == "cuda" and torch.cuda.is_available() and self.use_fp16:
            # Check if GPU supports BF16 (e.g., Ampere architecture or newer like A100, H100, RTX 30xx/40xx)
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                logger.info("CUDA GPU supports BF16. Selecting torch.bfloat16 for MiniCPM-V optimization.")
                return torch.bfloat16
            else:
                logger.info("BF16 not supported or older CUDA GPU. Selecting torch.float16 for MiniCPM-V.")
                return torch.float16
        elif self.device == "mps" and self.use_fp16:
            return torch.float16
        return torch.float32

    def load_model(self) -> None:
        """
        Load MiniCPM-V model weights and tokenizer into memory/VRAM.
        Applies automatic BF16/FP16 quantization for GPU acceleration.
        """
        if self.is_loaded:
            logger.debug(f"MiniCPMVModel '{self.model_name}' already loaded.")
            return

        logger.info(f"Loading MiniCPM-V model '{self.model_path}' onto {self.device}...")
        from transformers import AutoModel, AutoTokenizer

        target_dtype = self._get_target_dtype()

        # Load tokenizer and model with remote code trust required by MiniCPM-V
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=target_dtype
        )

        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(device=self.device, dtype=target_dtype)
        elif self.device != "cpu":
            self.model = self.model.to(device=self.device)

        self.model.eval()
        self.is_loaded = True
        logger.info(f"MiniCPMVModel '{self.model_name}' successfully loaded into VRAM.")

    def unload_model(self) -> None:
        """Unload MiniCPM-V model weights from memory and release CUDA cache."""
        if not self.is_loaded:
            return
        logger.info(f"Unloading MiniCPMVModel '{self.model_name}'...")
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_action(self, image: Image.Image, candidate_actions: List[str]) -> Dict[str, Any]:
        """
        Predict the best robot action from candidate action strings using MiniCPM-V visual reasoning.
        
        Args:
            image: PIL Image captured by robot camera.
            candidate_actions: List of text prompts (e.g., ['move forward', 'turn left', 'stop']).
            
        Returns:
            Dict[str, Any]: Dictionary containing best action, confidence, all scores, model used, and timing.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load_model()

        start_time = time.perf_counter()

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Construct concise prompt for deterministic action selection
        actions_str = ", ".join(f'"{act}"' for act in candidate_actions)
        prompt = (
            "You are an autonomous robot navigation assistant. Analyze the visual scene in this image and select the single most appropriate action "
            f"from the following candidate actions: [{actions_str}]. "
            "Reply ONLY with the exact chosen action string from the list without any extra words, explanation, quotes, or punctuation."
        )
        msgs = [{"role": "user", "content": prompt}]

        with torch.inference_mode():
            res, _, _ = self.model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=self.tokenizer,
                sampling=False,  # Greedy decoding for deterministic action selection
                temperature=1.0
            )

        res_str = str(res).strip()
        res_clean = res_str.strip('".\'').lower()

        best_action = candidate_actions[0] if candidate_actions else "stop"
        matched = False

        # Exact match attempt
        for act in candidate_actions:
            if act.strip().lower() == res_clean:
                best_action = act
                matched = True
                break

        # Substring match fallback
        if not matched:
            for act in candidate_actions:
                act_lower = act.strip().lower()
                if act_lower in res_clean or res_clean in act_lower:
                    best_action = act
                    matched = True
                    break

        # Calculate score distribution
        if candidate_actions:
            best_conf = 0.95 if matched else 0.70
            other_conf = (1.0 - best_conf) / max(1, len(candidate_actions) - 1) if len(candidate_actions) > 1 else 0.0
            all_scores = {act: (best_conf if act == best_action else other_conf) for act in candidate_actions}
        else:
            best_conf = 1.0
            all_scores = {best_action: 1.0}

        elapsed_ms = self.measure_time_ms(start_time)

        return {
            "action": best_action,
            "confidence": best_conf,
            "all_scores": all_scores,
            "model_used": self.model_name,
            "inference_time_ms": elapsed_ms,
            "raw_response": res_str
        }

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """Raise NotImplementedError as MiniCPM-V is configured as an action classification / VLM model."""
        raise NotImplementedError("MiniCPMVModel is an action classification / VLM model and does not support fast bounding box detection.")

    def generate_completion(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "Describe the visual contents of this image.",
        temperature: float = 0.7,
        max_tokens: int = 512,
        msgs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a Vision-Language completion or VQA response using MiniCPM-V.
        
        Args:
            image: Optional PIL Image to analyze.
            prompt: Text prompt / question for the visual scene.
            temperature: Sampling temperature (0.0 for greedy decoding).
            max_tokens: Maximum tokens to generate.
            msgs: Optional OpenAI-style messages list.
            
        Returns:
            Dict[str, Any]: Structured OpenAI-compatible completion response with latency metrics.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load_model()

        start_time = time.perf_counter()

        if image is not None and image.mode != "RGB":
            image = image.convert("RGB")

        if not msgs:
            msgs = [{"role": "user", "content": prompt}]

        with torch.inference_mode():
            res, _, _ = self.model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=self.tokenizer,
                sampling=(temperature > 0.0),
                temperature=max(0.1, temperature),
                max_new_tokens=max_tokens
            )

        res_str = str(res).strip()
        elapsed_ms = self.measure_time_ms(start_time)

        prompt_words = max(1, len(prompt.split()))
        comp_words = max(1, len(res_str.split()))

        return {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "text": res_str,
                    "message": {"role": "assistant", "content": res_str},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_words,
                "completion_tokens": comp_words,
                "total_tokens": prompt_words + comp_words
            },
            "inference_time_ms": round(elapsed_ms, 2),
            "raw_response": res_str
        }
