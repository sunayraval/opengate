"""
Mage-VL Model Implementation for Multimodal Zero-Shot Action Classification & Vision Reasoning.
Leverages microsoft/Mage-VL with automatic FP16/BF16 precision optimization.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
from app.models.base import BaseVisionModel

logger = logging.getLogger(__name__)


class MageVLModel(BaseVisionModel):
    """
    Mage-VL Vision-Language Model implementation.
    Uses Hugging Face pipeline with trust_remote_code=True for Microsoft's codec-native architecture.
    """

    def __init__(
        self,
        name: str = "microsoft/Mage-VL",
        model_path: str = "microsoft/Mage-VL",
        use_fp16: bool = True,
        device: Optional[str] = None
    ):
        super().__init__(name=name, model_type="vision_language", device=device)
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.pipe = None
        
    def _get_target_dtype(self) -> torch.dtype:
        if self.device == "cuda" and torch.cuda.is_available() and self.use_fp16:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            else:
                return torch.float16
        return torch.float32

    def load_model(self) -> None:
        if self.is_loaded:
            return

        logger.info(f"Loading Mage-VL model '{self.model_path}' onto {self.device}...")
        
        # --- MONKEY PATCH FOR MICROSOFT REMOTE CODE BUG ---
        # Microsoft's code uses 'PreTrainedConfig' (capital T), but newer transformers
        # renamed it to 'PretrainedConfig' (lowercase t). We dynamically patch it here.
        import transformers.configuration_utils
        if not hasattr(transformers.configuration_utils, 'PreTrainedConfig'):
            transformers.configuration_utils.PreTrainedConfig = getattr(transformers.configuration_utils, 'PretrainedConfig', None)
            
        # Microsoft's code uses @strict from huggingface_hub on regular classes,
        # which crashes in newer huggingface-hub versions. We mock it out.
        import huggingface_hub.dataclasses
        def dummy_strict(cls=None):
            if cls is None:
                return lambda c: c
            return cls
        huggingface_hub.dataclasses.strict = dummy_strict
        # --------------------------------------------------
        
        from transformers import pipeline
        
        target_dtype = self._get_target_dtype()
        
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_path,
            trust_remote_code=True,
            device=self.device,
            torch_dtype=target_dtype
        )
        self.is_loaded = True
        logger.info(f"MageVLModel '{self.model_name}' successfully loaded.")

    def unload_model(self) -> None:
        if not self.is_loaded:
            return
        logger.info(f"Unloading MageVLModel '{self.model_name}'...")
        self.pipe = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_action(self, image: Image.Image, candidate_actions: List[str]) -> Dict[str, Any]:
        raise NotImplementedError("Action prediction directly not implemented. Use generate_completion.")

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        raise NotImplementedError("Detection not supported by MageVLModel.")

    def generate_completion(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "Describe the visual contents of this image.",
        temperature: float = 0.7,
        max_tokens: int = 512,
        msgs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        
        if not self.is_loaded or self.pipe is None:
            self.load_model()
            
        start_time = time.perf_counter()

        if image is not None and image.mode != "RGB":
            image = image.convert("RGB")

        # Mage-VL specific message structure
        if not msgs:
            content = []
            if image is not None:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            msgs = [{"role": "user", "content": content}]

        # Set up generation arguments
        gen_kwargs = {
            "max_new_tokens": max_tokens,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(0.1, temperature)
        else:
            gen_kwargs["do_sample"] = False

        if image is not None:
            result = self.pipe(image, prompt=msgs, generate_kwargs=gen_kwargs)
        else:
            result = self.pipe(prompt=msgs, generate_kwargs=gen_kwargs)
            
        # Parse result - pipeline output structure depends on model, typically [{'generated_text': ...}]
        res_str = ""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'generated_text' in result[0]:
                res_str = result[0]['generated_text']
            elif isinstance(result[0], str):
                res_str = result[0]
            else:
                res_str = str(result[0])
        else:
            res_str = str(result)
            
        # Optional: remove the system/user prompt if the pipeline returns the full text
        # (Usually pipeline("image-text-to-text") just returns the newly generated text)
        res_str = res_str.strip()

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
