"""
Base Vision Model definition for the Real-Time AI CV Robot Framework.
Provides an abstract base class with standardized properties, timing utilities,
and interface methods for action prediction and object detection models.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
from contextlib import contextmanager


@contextmanager
def inference_timer():
    """
    Context manager to accurately measure inference execution time in milliseconds.
    
    Yields:
        dict: A dictionary containing 'elapsed_ms' which is updated when the context exits.
    """
    start_time = time.perf_counter()
    timer_data = {"elapsed_ms": 0.0}
    try:
        yield timer_data
    finally:
        timer_data["elapsed_ms"] = (time.perf_counter() - start_time) * 1000.0


class BaseVisionModel(ABC):
    """
    Abstract base class for all computer vision models in the robot framework.
    
    Attributes:
        model_name (str): Unique identifier name of the model.
        model_type (str): Type of model, either 'action' (classification/VLM) or 'detection'.
        device (str): Device where the model resides ('cuda' or 'cpu').
        is_loaded (bool): Status indicating whether model weights are currently in VRAM/memory.
    """

    def __init__(self, name: str, model_type: str, device: Optional[str] = None):
        """
        Initialize the base vision model.
        
        Args:
            name: Unique name identifier for the model.
            model_type: The operational type of model ('action' or 'detection').
            device: Target computing device ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.model_name: str = name
        self.model_type: str = model_type
        if device is None:
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device: str = device
        self.is_loaded: bool = False

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model weights into memory/VRAM and apply optimizations (e.g., FP16 half-precision).
        Must update self.is_loaded to True upon completion.
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Unload model weights from memory/VRAM and release GPU resources.
        Must update self.is_loaded to False upon completion.
        """
        pass

    @abstractmethod
    def predict_action(self, image: Image.Image, candidate_actions: List[str]) -> Dict[str, Any]:
        """
        Predict the best action from candidate action strings given an input image.
        
        Args:
            image: PIL Image captured from robot camera or input source.
            candidate_actions: List of text prompts representing potential actions.
            
        Returns:
            Dict[str, Any]: Structured result containing:
                - 'action': Best candidate action string.
                - 'confidence': Confidence score of the selected action.
                - 'all_scores': Dictionary mapping all candidate actions to their scores.
                - 'model_used': Name of the model that performed inference.
                - 'inference_time_ms': Execution duration in milliseconds.
        """
        pass

    @abstractmethod
    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect objects and bounding boxes within an input image.
        
        Args:
            image: PIL Image captured from robot camera or input source.
            
        Returns:
            Dict[str, Any]: Structured result containing:
                - 'bounding_boxes': List of detected object bounding boxes and classes.
                - 'model_used': Name of the model that performed inference.
                - 'inference_time_ms': Execution duration in milliseconds.
        """
        pass

    def measure_time_ms(self, start_time: float) -> float:
        """
        Helper method to compute elapsed time in milliseconds from a starting timestamp.
        
        Args:
            start_time: Timestamp obtained via time.perf_counter().
            
        Returns:
            float: Elapsed duration in milliseconds.
        """
        return (time.perf_counter() - start_time) * 1000.0

    def generate_completion(
        self,
        image: Optional[Image.Image] = None,
        prompt: str = "Describe the visual contents of this image.",
        temperature: float = 0.7,
        max_tokens: int = 512,
        msgs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a multimodal Vision-Language AI completion response.
        Subclasses like MiniCPMVModel override this with generative LLM inference.
        Default fallback uses classification or detection output formatted as text.
        """
        start_time = time.perf_counter()
        if not self.is_loaded:
            self.load_model()
            
        res_str = f"[Vision Model {self.model_name}] Visual reasoning completion not natively supported by this model class."
        if image and self.model_type == "action" and hasattr(self, "predict_action"):
            try:
                pred = self.predict_action(image, ["move forward", "turn left", "turn right", "stop", "proceed safely"])
                res_str = f"Based on visual analysis with {self.model_name}, the recommended navigation action is '{pred.get('action')}' with {pred.get('confidence', 0)*100:.1f}% confidence."
            except Exception:
                pass
        elif image and self.model_type == "detection" and hasattr(self, "detect"):
            try:
                det = self.detect(image)
                boxes = det.get("bounding_boxes", [])
                classes = [b.get("class_name", "object") if isinstance(b, dict) else getattr(b, "class_name", "object") for b in boxes]
                res_str = f"Detected {len(boxes)} objects in scene using {self.model_name}: {', '.join(set(classes)) if classes else 'no objects detected'}."
            except Exception:
                pass

        elapsed_ms = self.measure_time_ms(start_time)
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
                "prompt_tokens": max(1, len(prompt.split())),
                "completion_tokens": max(1, len(res_str.split())),
                "total_tokens": max(1, len(prompt.split())) + max(1, len(res_str.split()))
            },
            "inference_time_ms": round(elapsed_ms, 2),
            "raw_response": res_str
        }
