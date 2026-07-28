"""
AI/CV Multi-Model Inference Engine Package for Real-Time AI CV Robot Framework.
Exports base models, model implementations, and the thread-safe ModelRegistry singleton.
"""

from app.models.base import BaseVisionModel
from app.models.registry import ModelRegistry, model_registry
from app.models.clip_model import OpenCLIPModel
from app.models.detector_model import YOLODetectorModel
from app.models.minicpm_model import MiniCPMVModel

__all__ = [
    "BaseVisionModel",
    "ModelRegistry",
    "model_registry",
    "OpenCLIPModel",
    "YOLODetectorModel",
    "MiniCPMVModel",
]
