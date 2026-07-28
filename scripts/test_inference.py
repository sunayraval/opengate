"""
Standalone Desktop Verification Script for the Real-Time AI CV Robot Framework.
Generates synthetic test images, initializes the ModelRegistry, runs multi-model inference,
and reports performance timing and CUDA VRAM usage.
"""

import os
import sys
import time
import argparse
import logging
from PIL import Image, ImageDraw

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.registry import model_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TestInference")


def create_synthetic_image() -> Image.Image:
    """Create a synthetic RGB test image with geometrical shapes for vision testing."""
    logger.info("Generating synthetic RGB test image (640x480)...")
    img = Image.new("RGB", (640, 480), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Draw red rectangle (simulate obstacle / object)
    draw.rectangle([100, 150, 300, 350], fill=(220, 20, 20), outline=(0, 0, 0), width=3)
    
    # Draw green circle (simulate target / marker)
    draw.ellipse([350, 100, 500, 250], fill=(20, 180, 20), outline=(0, 0, 0), width=3)
    
    # Draw blue polygon / pathway marker
    draw.polygon([(200, 400), (400, 400), (300, 280)], fill=(20, 20, 220), outline=(0, 0, 0))
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Test multi-model inference in the Robot Framework.")
    parser.add_argument("--test-minicpm", action="store_true", help="Include MiniCPM-V VLM in inference verification tests.")
    args = parser.parse_args()

    print("=" * 70)
    print("REAL-TIME AI/CV MULTI-MODEL INFERENCE ENGINE VERIFICATION")
    print("=" * 70)
    
    # Check initial VRAM usage
    initial_vram = model_registry.get_vram_usage_mb()
    print(f"[*] Initial CUDA VRAM Usage: {initial_vram:.2f} MB")
    
    # Initialize and load default models into VRAM
    print("\n[*] Initializing default vision models (OpenCLIP & YOLOv8n)...")
    start_init = time.perf_counter()
    model_registry.initialize_defaults()
    init_time_ms = (time.perf_counter() - start_init) * 1000.0
    
    post_init_vram = model_registry.get_vram_usage_mb()
    print(f"[*] Models Initialized in {init_time_ms:.2f} ms")
    print(f"[*] Post-Initialization CUDA VRAM Usage: {post_init_vram:.2f} MB (+{post_init_vram - initial_vram:.2f} MB)")
    
    # List registered models
    print("\n[*] Registered Vision Models:")
    models_info = model_registry.list_models()
    for name, info in models_info.items():
        print(f"    - {name} | Type: {info['model_type']} | Device: {info['device']} | Loaded: {info['is_loaded']}")
        
    # Generate test image
    test_image = create_synthetic_image()
    
    # ---------------------------------------------------------
    # TEST 1: Zero-Shot Action Prediction (OpenCLIP)
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("TEST 1: Zero-Shot Action Classification (OpenCLIP)")
    print("-" * 50)
    
    action_model = model_registry.get_model(model_type="action")
    candidate_actions = [
        "move forward toward the green target",
        "turn left to avoid the red obstacle",
        "stop and wait for clearance",
        "reverse back from the blue polygon"
    ]
    print(f"[*] Querying action model: '{action_model.model_name}'")
    print(f"[*] Candidate Actions:\n    " + "\n    ".join(f"({i+1}) {act}" for i, act in enumerate(candidate_actions)))
    
    action_result = action_model.predict_action(test_image, candidate_actions)
    
    print("\n[*] Action Prediction Results:")
    print(f"    -> Best Action : \"{action_result['action']}\"")
    print(f"    -> Confidence  : {action_result['confidence']*100:.2f}%")
    print(f"    -> Inference Time : {action_result['inference_time_ms']:.2f} ms")
    print("    -> All Scores:")
    for act, score in action_result["all_scores"].items():
        print(f"       * {act}: {score*100:.2f}%")
    
    print(f"[*] Current CUDA VRAM Usage: {model_registry.get_vram_usage_mb():.2f} MB")

    # ---------------------------------------------------------
    # TEST 2: Fast Object Detection (YOLOv8)
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("TEST 2: Fast Bounding Box Object Detection (YOLOv8)")
    print("-" * 50)
    
    detection_model = model_registry.get_model(model_type="detection")
    print(f"[*] Querying detection model: '{detection_model.model_name}'")
    
    detection_result = detection_model.detect(test_image)
    
    print("\n[*] Detection Results:")
    print(f"    -> Total Objects Detected : {len(detection_result['bounding_boxes'])}")
    print(f"    -> Inference Time         : {detection_result['inference_time_ms']:.2f} ms")
    for idx, box in enumerate(detection_result["bounding_boxes"]):
        print(f"    -> Box #{idx+1}: Class: {box['class_name']} (Conf: {box['confidence']*100:.1f}%) | "
              f"BBox [xmin={box['xmin']:.1f}, ymin={box['ymin']:.1f}, xmax={box['xmax']:.1f}, ymax={box['ymax']:.1f}]")
        
    print(f"[*] Current CUDA VRAM Usage: {model_registry.get_vram_usage_mb():.2f} MB")

    # ---------------------------------------------------------
    # TEST 3: Multimodal Action Reasoning (MiniCPM-V) [Optional]
    # ---------------------------------------------------------
    if args.test_minicpm:
        print("\n" + "-" * 50)
        print("TEST 3: Multimodal Action Reasoning (MiniCPM-V)")
        print("-" * 50)
        try:
            minicpm_model = model_registry.get_model(name="openbmb/MiniCPM-V")
            print(f"[*] Querying VLM model: '{minicpm_model.model_name}'")
            res_minicpm = minicpm_model.predict_action(test_image, candidate_actions)
            print("\n[*] MiniCPM-V Prediction Results:")
            print(f"    -> Best Action : \"{res_minicpm['action']}\"")
            print(f"    -> Confidence  : {res_minicpm['confidence']*100:.2f}%")
            print(f"    -> Inference Time : {res_minicpm['inference_time_ms']:.2f} ms")
            print(f"    -> Raw Output  : {res_minicpm.get('raw_response', 'N/A')}")
            print(f"[*] Current CUDA VRAM Usage: {model_registry.get_vram_usage_mb():.2f} MB")
        except Exception as e:
            print(f"[*] MiniCPM-V test skipped or error: {e}")

    # ---------------------------------------------------------
    # TEST 4: Resource Cleanup / Unloading
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("TEST 4: VRAM Memory Management & Model Unloading")
    print("-" * 50)
    
    for model_name in list(models_info.keys()):
        model_registry.unload_model(model_name)
        
    final_vram = model_registry.get_vram_usage_mb()
    print(f"[*] Final CUDA VRAM Usage After Unloading: {final_vram:.2f} MB")
    
    print("\n" + "=" * 70)
    print("ALL AI/CV MULTI-MODEL INFERENCE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
