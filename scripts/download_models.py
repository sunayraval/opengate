"""
Model Pre-Caching CLI Script for Real-Time AI CV Robot Framework.
Downloads and caches OpenCLIP and YOLOv8 weights offline to prevent network latency at startup.
"""

import os
import argparse
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_clip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", cache_dir: str = "./weights/clip"):
    """
    Download and cache OpenCLIP model weights and tokenizer offline.
    
    Args:
        model_name: OpenCLIP model architecture name.
        pretrained: Pretrained dataset name.
        cache_dir: Target local directory for caching weight artifacts.
    """
    import open_clip
    logger.info(f"Downloading OpenCLIP model '{model_name}' ({pretrained}) to cache directory: {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download weights and transforms
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        cache_dir=cache_dir
    )
    # Download tokenizer
    tokenizer = open_clip.get_tokenizer(model_name)
    logger.info("OpenCLIP model and tokenizer successfully downloaded and cached!")


def download_yolo(model_name: str = "yolov8n.pt", target_dir: str = "./weights"):
    """
    Download and cache YOLOv8 weights offline.
    
    Args:
        model_name: Name of YOLO model weight file.
        target_dir: Target directory for storing weight files.
    """
    from ultralytics import YOLO
    import shutil
    
    logger.info(f"Downloading YOLO detector weights '{model_name}'...")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)
    
    # Instantiate YOLO to trigger automatic download
    model = YOLO(model_name)
    
    # If downloaded to current working directory, copy/move to target_dir if needed
    if os.path.exists(model_name) and os.path.abspath(model_name) != os.path.abspath(target_path):
        logger.info(f"Moving '{model_name}' to target directory '{target_path}'...")
        shutil.copy2(model_name, target_path)
    logger.info(f"YOLO detector weights successfully cached at '{target_path}'!")


def download_minicpm(model_name: str = "openbmb/MiniCPM-V", cache_dir: str = "./weights/minicpm"):
    """
    Download and cache MiniCPM-V model weights and tokenizer offline.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'openbmb/MiniCPM-V').
        cache_dir: Target local directory for storing weight artifacts.
    """
    from transformers import AutoModel, AutoTokenizer
    logger.info(f"Downloading MiniCPM-V model '{model_name}' to cache directory: {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    # Download weights
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
    logger.info("MiniCPM-V model and tokenizer successfully downloaded and cached!")


def main():
    parser = argparse.ArgumentParser(description="Pre-cache AI/CV models for offline usage in the Robot Framework.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["clip", "yolo", "minicpm", "all"],
        default="all",
        help="Specify which model weights to download and cache (clip, yolo, minicpm, or all)."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download and cache all supported vision models."
    )
    args = parser.parse_args()

    target_model = "all" if args.all else args.model
    logger.info(f"Starting offline weight caching for: {target_model}")

    if target_model in ["clip", "all"]:
        try:
            download_clip()
        except Exception as e:
            logger.error(f"Failed to download OpenCLIP model: {e}", exc_info=True)

    if target_model in ["yolo", "all"]:
        try:
            download_yolo()
        except Exception as e:
            logger.error(f"Failed to download YOLO model: {e}", exc_info=True)

    if target_model in ["minicpm", "all"]:
        try:
            download_minicpm()
        except Exception as e:
            logger.error(f"Failed to download MiniCPM-V model: {e}", exc_info=True)

    logger.info("Model pre-caching operations complete.")


if __name__ == "__main__":
    main()
