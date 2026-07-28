"""
OpenCLIP Model Implementation for Zero-Shot Action Classification.
Provides high-performance inference with FP16 half-precision optimization on CUDA devices.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
import open_clip
from app.models.base import BaseVisionModel

logger = logging.getLogger(__name__)


class OpenCLIPModel(BaseVisionModel):
    """
    OpenCLIP model implementation for zero-shot action classification in robot navigation/control.
    Supports FP16 half-precision on CUDA for 2x faster inference and 50% less VRAM usage.
    """

    def __init__(
        self,
        name: str = "clip-vit-base-patch32",
        clip_model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        use_fp16: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize OpenCLIP action classification model.
        
        Args:
            name: Registry identifier for this model instance.
            clip_model_name: Architecture name in open_clip (e.g., 'ViT-B-32').
            pretrained: Pretrained weight dataset name (e.g., 'laion2b_s34b_b79k').
            use_fp16: Whether to cast model to half precision on CUDA.
            device: Target computation device ('cuda' or 'cpu').
        """
        super().__init__(name=name, model_type="action", device=device)
        self.clip_model_name = clip_model_name
        self.pretrained = pretrained
        self.use_fp16 = use_fp16
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load_model(self) -> None:
        """
        Load OpenCLIP model and transforms into VRAM/memory.
        Converts weights to FP16 if running on a CUDA device with FP16 enabled.
        """
        if self.is_loaded:
            logger.debug(f"OpenCLIPModel '{self.model_name}' already loaded.")
            return

        logger.info(f"Loading OpenCLIP model '{self.clip_model_name}' ({self.pretrained}) onto {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name,
            pretrained=self.pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.clip_model_name)

        self.model.eval()
        if self.device == "cuda" and self.use_fp16 and torch.cuda.is_available():
            logger.info(f"Converting OpenCLIP model '{self.model_name}' to FP16 half precision for CUDA optimization...")
            self.model = self.model.half()

        self.is_loaded = True
        logger.info(f"OpenCLIPModel '{self.model_name}' successfully loaded into VRAM.")

    def unload_model(self) -> None:
        """Unload OpenCLIP model weights from memory and release CUDA cache."""
        if not self.is_loaded:
            return
        logger.info(f"Unloading OpenCLIP model '{self.model_name}'...")
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_action(self, image: Image.Image, candidate_actions: List[str]) -> Dict[str, Any]:
        """
        Predict the best robot action from candidate action strings using zero-shot classification.
        
        Args:
            image: PIL Image captured by robot camera.
            candidate_actions: List of text prompts (e.g., ['move forward', 'turn left', 'stop']).
            
        Returns:
            Dict[str, Any]: Dictionary containing best action, confidence, all scores, model used, and timing.
        """
        if not self.is_loaded or self.model is None or self.preprocess is None or self.tokenizer is None:
            self.load_model()

        start_time = time.perf_counter()

        # Preprocess input image and move to target device
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        if self.device == "cuda" and self.use_fp16 and torch.cuda.is_available():
            image_tensor = image_tensor.half()

        # Tokenize candidate action text strings
        text_tokens = self.tokenizer(candidate_actions).to(self.device)

        with torch.inference_mode():
            # Compute normalized image and text embeddings
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity and scale by logit_scale
            logit_scale = self.model.logit_scale.exp() if hasattr(self.model, "logit_scale") else 100.0
            logits = (logit_scale * image_features @ text_features.T).squeeze(0)
            
            # Softmax probabilities over candidate actions
            probs = logits.softmax(dim=-1).cpu().to(torch.float32).numpy()

        all_scores = {action: float(prob) for action, prob in zip(candidate_actions, probs)}
        best_idx = int(probs.argmax())
        best_action = candidate_actions[best_idx]
        best_confidence = float(probs[best_idx])
        
        elapsed_ms = self.measure_time_ms(start_time)

        return {
            "action": best_action,
            "confidence": best_confidence,
            "all_scores": all_scores,
            "model_used": self.model_name,
            "inference_time_ms": elapsed_ms
        }

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """Raise NotImplementedError as OpenCLIP is strictly an action classification model."""
        raise NotImplementedError("OpenCLIPModel is an action classification model and does not support object detection.")
