"""
Model Registry for the Real-Time AI CV Robot Framework.
Provides a thread-safe singleton registry that manages loading, unloading,
VRAM usage tracking, and default model initialization.
"""

import threading
import logging
import torch
from typing import Dict, Any, Optional
from app.models.base import BaseVisionModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Thread-safe singleton registry for managing multi-model computer vision inference engines.
    Manages loading, unloading, VRAM usage tracking, and automatic default initialization.
    """
    _instance: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        with self._lock:
            if getattr(self, "_initialized", False):
                return
            self.models: Dict[str, BaseVisionModel] = {}
            self.default_completion_model: Optional[str] = None
            self._model_lock = threading.Lock()
            self._initialized = True

    def register_model(self, name: str, model_instance: BaseVisionModel) -> None:
        """
        Register a model instance in the registry.
        
        Args:
            name: Unique identifier name for the model.
            model_instance: An instance inheriting from BaseVisionModel.
        """
        with self._model_lock:
            self.models[name] = model_instance
            if model_instance.model_type == "vision_language" and self.default_completion_model is None:
                self.default_completion_model = name
            logger.info(f"Registered model: {name} (type: {model_instance.model_type}, device: {model_instance.device})")

    def load_model(self, name: str) -> None:
        """
        Load a specific model into VRAM/memory if not already loaded.
        
        Args:
            name: Name of the registered model to load.
        """
        with self._model_lock:
            if name not in self.models:
                raise KeyError(f"Model '{name}' is not registered in ModelRegistry.")
            model = self.models[name]
            if not model.is_loaded:
                logger.info(f"Loading model '{name}' into {model.device}...")
                model.load_model()
                logger.info(f"Model '{name}' loaded successfully.")
            else:
                logger.debug(f"Model '{name}' is already loaded.")

    def unload_model(self, name: str) -> None:
        """
        Unload a specific model from VRAM/memory and free GPU memory cache.
        
        Args:
            name: Name of the registered model to unload.
        """
        with self._model_lock:
            if name not in self.models:
                raise KeyError(f"Model '{name}' is not registered in ModelRegistry.")
            model = self.models[name]
            if model.is_loaded:
                logger.info(f"Unloading model '{name}'...")
                model.unload_model()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Model '{name}' unloaded successfully and VRAM cache cleared.")

    def get_model(self, name: Optional[str] = None, model_type: str = "action") -> BaseVisionModel:
        """
        Retrieve a model by name. If name is None, returns default model for the requested model_type.
        Automatically loads the model into memory/VRAM if it is registered but not currently loaded.
        
        Args:
            name: Name of the model to retrieve. Defaults to standard action/detection model if None.
            model_type: Expected model type ('action' or 'detection').
            
        Returns:
            BaseVisionModel: The requested and loaded model instance.
        """
        target_name = name
        if target_name is None:
            target_name = self.default_completion_model

        if target_name is None or target_name not in self.models:
            if target_name is not None and target_name not in self.models:
                logger.info(f"Model '{target_name}' not in registry. Attempting dynamic initialization...")
                from app.models.clip_model import OpenCLIPModel
                from app.models.detector_model import YOLODetectorModel
                from app.models.hf_vlm_model import HuggingFaceVLM

                name_lower = target_name.lower()
                if "mage-vl" in name_lower:
                    new_model = HuggingFaceVLM(name=target_name, model_path="microsoft/Phi-3.5-vision-instruct")
                elif "minicpm" in name_lower or "openbmb" in name_lower:
                    new_model = HuggingFaceVLM(name=target_name, model_path="openbmb/MiniCPM-V")
                elif "yolo" in name_lower or target_name.endswith(".pt"):
                    new_model = YOLODetectorModel(name=target_name, weights_path=target_name)
                else:
                    new_model = OpenCLIPModel(name=target_name, clip_model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
                self.register_model(target_name, new_model)
            else:
                raise KeyError(f"No valid model found for name='{target_name}', model_type='{model_type}'. Please register or initialize default models first.")

        # Ensure model is loaded
        model = self.models[target_name]
        if not model.is_loaded:
            self.load_model(target_name)

        return model

    def list_models(self) -> Dict[str, Any]:
        """
        Return a dictionary of all registered models, their type, device, and load status.
        
        Returns:
            Dict[str, Any]: Mapping of model names to metadata dictionaries.
        """
        with self._model_lock:
            result = {}
            for name, model in self.models.items():
                result[name] = {
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "device": model.device,
                    "is_loaded": model.is_loaded
                }
            return result

    def initialize_defaults(self) -> None:
        """
        Instantiate and register default completion model (MiniCPM-V),
        and pre-load it into VRAM so the server is instantly warm on startup.
        """
        from app.models.hf_vlm_model import HuggingFaceVLM

        logger.info("Initializing default completion model...")
        default_model = HuggingFaceVLM(name="openbmb/MiniCPM-V", model_path="openbmb/MiniCPM-V")
        self.register_model(default_model.model_name, default_model)
        self.default_completion_model = default_model.model_name
        
        # Pre-load to VRAM immediately
        self.load_model(default_model.model_name)
        
        logger.info("Default completion model initialized and loaded into VRAM.")

    def get_vram_usage_mb(self) -> float:
        """
        Return current CUDA VRAM allocation in megabytes (MB) if CUDA is available, else 0.0.
        
        Returns:
            float: Megabytes of allocated GPU memory.
        """
        if torch.cuda.is_available():
            allocated_bytes = torch.cuda.memory_allocated()
            return float(allocated_bytes) / (1024.0 * 1024.0)
        return 0.0


# Global singleton instance
model_registry = ModelRegistry()
