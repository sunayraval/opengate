"""
YOLO Object Detector Model Implementation for Real-Time Detection.
Uses Ultralytics YOLOv8 for fast bounding box detection with half-precision support on GPU.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
from ultralytics import YOLO
from app.models.base import BaseVisionModel

logger = logging.getLogger(__name__)


class YOLODetectorModel(BaseVisionModel):
    """
    YOLOv8 object detector model implementation for robot vision perception.
    Runs fast bounding box detection with optional half-precision (FP16) on CUDA GPUs.
    """

    def __init__(
        self,
        name: str = "yolov8n.pt",
        weights_path: str = "yolov8n.pt",
        use_fp16: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO detector model.
        
        Args:
            name: Registry identifier name for this model instance.
            weights_path: Path or filename for YOLO weight file (e.g., 'yolov8n.pt').
            use_fp16: Whether to use FP16 half-precision inference on CUDA.
            device: Target computing device ('cuda' or 'cpu').
        """
        super().__init__(name=name, model_type="detection", device=device)
        self.weights_path = weights_path
        self.use_fp16 = use_fp16
        self.model = None

    def load_model(self) -> None:
        """Load YOLO model weights into memory/VRAM and move to CUDA if available."""
        if self.is_loaded:
            logger.debug(f"YOLODetectorModel '{self.model_name}' already loaded.")
            return

        logger.info(f"Loading YOLO detector weights '{self.weights_path}' onto {self.device}...")
        self.model = YOLO(self.weights_path)
        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")

        self.is_loaded = True
        logger.info(f"YOLODetectorModel '{self.model_name}' successfully loaded into VRAM.")

    def unload_model(self) -> None:
        """Unload YOLO model weights from memory and clear CUDA memory cache."""
        if not self.is_loaded:
            return
        logger.info(f"Unloading YOLODetectorModel '{self.model_name}'...")
        self.model = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run object detection on the input image using YOLOv8.
        
        Args:
            image: PIL Image captured from robot camera.
            
        Returns:
            Dict[str, Any]: Structured result containing bounding boxes, model used, and timing.
        """
        if not self.is_loaded or self.model is None:
            self.load_model()

        start_time = time.perf_counter()

        # Run fast YOLO inference with half-precision support on CUDA
        use_half = (self.device == "cuda" and self.use_fp16 and torch.cuda.is_available())
        results = self.model.predict(
            image,
            device=self.device,
            half=use_half,
            verbose=False
        )

        bounding_boxes = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                class_name = result.names[cls_id] if hasattr(result, "names") and cls_id in result.names else str(cls_id)

                bounding_boxes.append({
                    "xmin": float(xyxy[0]),
                    "ymin": float(xyxy[1]),
                    "xmax": float(xyxy[2]),
                    "ymax": float(xyxy[3]),
                    "confidence": conf,
                    "class_name": class_name
                })

        elapsed_ms = self.measure_time_ms(start_time)

        return {
            "bounding_boxes": bounding_boxes,
            "model_used": self.model_name,
            "inference_time_ms": elapsed_ms
        }

    def predict_action(self, image: Image.Image, candidate_actions: List[str]) -> Dict[str, Any]:
        """Raise NotImplementedError as YOLO is strictly an object detection model."""
        raise NotImplementedError("YOLODetectorModel is an object detection model and does not support action classification.")
