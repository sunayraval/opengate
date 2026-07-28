#!/usr/bin/env python3
"""
Real-Time AI CV Robot Client for Raspberry Pi 5
=================================================
This script runs on your Raspberry Pi 5. It connects to your camera, compresses frames
to lightweight JPEG buffers (~25KB), and sends them over WAN/Internet to your Windows Desktop
running the FastAPI / Cloudflare Tunnel AI serving framework.

Usage Examples:
  # 1. Zero-shot action decision (Default OpenCLIP model):
  python robot_client.py --url "https://api.myrobot.com" --mode action --actions "move forward,turn left,turn right,stop"

  # 2. Object detection (YOLOv8 bounding boxes):
  python robot_client.py --url "https://api.myrobot.com" --mode detect --fps 10

  # 3. Quick tunnel test with USB Camera index 0:
  python robot_client.py --url "https://xxxx.trycloudflare.com" --camera 0
"""

import argparse
import time
import json
import sys
import cv2
import requests

def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Client for Remote AI CV Framework")
    parser.add_argument("--url", type=str, required=True, help="Public HTTPS URL of your Desktop (e.g. https://api.myrobot.com or trycloudflare URL)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0 for USB/PiCam) or video path")
    parser.add_argument("--mode", type=str, choices=["action", "detect", "completion"], default="action", help="Inference mode: 'action' for zero-shot decision, 'detect' for YOLO bounding boxes, 'completion' for VLM chat/VQA")
    parser.add_argument("--actions", type=str, default="move forward,turn left to avoid obstacle,turn right to avoid obstacle,stop for obstacle", help="Comma-separated candidate action prompts for zero-shot decision")
    parser.add_argument("--prompt", type=str, default="Describe the scene and any potential hazards ahead.", help="Text prompt / question for completion mode")
    parser.add_argument("--model", type=str, default=None, help="Specific model name on server to query (optional, defaults to server default)")
    parser.add_argument("--fps", type=float, default=5.0, help="Target frames per second to send to remote server (default 5 FPS)")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG compression quality 1-100 (default 75, ~25KB payload)")
    return parser.parse_args()


# =====================================================================
# 🤖 AI RESPONSE & CUSTOM ACTION HOOKS
# =====================================================================
def on_ai_action_received(action: str, confidence: float, all_scores: dict, latency_ms: float):
    """
    HOOK: Process zero-shot navigation and decision-making predictions.
    Use this to trigger alerts, log state, or control custom logic based on AI decisions.
    """
    print(f"[{latency_ms:5.1f}ms] 🎯 DECISION: '{action}' (Confidence: {confidence*100:5.1f}%)")


def on_object_detections_received(bounding_boxes: list, latency_ms: float):
    """
    HOOK: Process real-time object detection bounding boxes.
    Each box is a dict: {'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class_name'}
    """
    print(f"[{latency_ms:5.1f}ms] 📦 DETECTED {len(bounding_boxes)} objects:")
    for box in bounding_boxes[:3]:  # Print top 3
        print(f"   -> {box['class_name']} ({box['confidence']*100:.1f}%) at [{box['xmin']:.1f}, {box['ymin']:.1f}, {box['xmax']:.1f}, {box['ymax']:.1f}]")
    if len(bounding_boxes) > 3:
        print(f"   ... and {len(bounding_boxes)-3} more.")


def on_ai_completion_received(text: str, model_used: str, latency_ms: float):
    """
    HOOK: Process multimodal Vision-Language completion and VQA responses.
    Use this to log visual summaries, speech synthesis (TTS), or complex scene analysis!
    """
    print(f"[{latency_ms:5.1f}ms] 💬 AI COMPLETION ({model_used}):")
    print(f"   \"{text}\"")


# =====================================================================
# 🌐 WAN CLIENT INFERENCE ENGINE
# =====================================================================
def run_client_loop(args):
    # Ensure URL doesn't end with trailing slash
    base_url = args.url.rstrip('/')
    
    print(f"🚀 Initializing Raspberry Pi 5 Camera Client...")
    print(f"📡 Remote AI Server : {base_url}")
    print(f"📷 Camera Index     : {args.camera}")
    print(f"⚡ Target FPS       : {args.fps} FPS")
    print(f"🗜️ JPEG Quality     : {args.jpeg_quality}")
    
    # 1. Test connection to server health endpoint
    health_url = f"{base_url}/health"
    try:
        print(f"🔍 Testing connection to {health_url}...")
        resp = requests.get(health_url, timeout=5.0)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Connected to AI Server! Status: {data.get('status')}")
            print(f"💻 Server GPU Available: {data.get('gpu_available')} ({data.get('gpu_name', 'N/A')})")
            print(f"🧠 Loaded Models: {data.get('loaded_models')}")
        else:
            print(f"⚠️ Server returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"❌ Could not connect to remote AI Server: {e}")
        print(f"   Please check that the Cloudflare tunnel is running on your desktop and the URL is correct.")
        sys.exit(1)

    # 2. Open Video Capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera}. Please check camera permissions or USB connection.")
        sys.exit(1)
        
    # Set camera resolution (VGA 640x480 is optimal for speed over WAN while preserving AI accuracy)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_interval = 1.0 / max(0.1, args.fps)
    candidate_actions_list = [a.strip() for a in args.actions.split(',') if a.strip()]

    print(f"\n🟢 STARTING REAL-TIME CONTROL LOOP (Press Ctrl+C to stop)...")
    if args.mode == "action":
        print(f"📋 Candidate Actions: {candidate_actions_list}")

    try:
        while True:
            loop_start = time.perf_counter()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Warning: Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            # Compress frame to JPEG in memory
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
            if not success:
                continue

            image_bytes = encoded_image.tobytes()
            payload_kb = len(image_bytes) / 1024.0

            # Send over WAN to Desktop
            send_start = time.perf_counter()
            files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
            
            try:
                if args.mode == "action":
                    endpoint = f"{base_url}/api/v1/infer/action"
                    data = {"candidate_actions": json.dumps(candidate_actions_list)}
                    if args.model:
                        data["model_name"] = args.model
                        
                    resp = requests.post(endpoint, files=files, data=data, timeout=3.0)
                    total_latency_ms = (time.perf_counter() - send_start) * 1000.0
                    
                    if resp.status_code == 200:
                        res_json = resp.json()
                        on_ai_action_received(
                            action=res_json.get("action", "UNKNOWN"),
                            confidence=res_json.get("confidence", 0.0),
                            all_scores=res_json.get("all_scores", {}),
                            latency_ms=total_latency_ms
                        )
                    else:
                        print(f"⚠️ API Error ({resp.status_code}): {resp.text[:100]}")

                elif args.mode == "detect":
                    endpoint = f"{base_url}/api/v1/infer/detect"
                    data = {}
                    if args.model:
                        data["model_name"] = args.model
                        
                    resp = requests.post(endpoint, files=files, data=data, timeout=3.0)
                    total_latency_ms = (time.perf_counter() - send_start) * 1000.0
                    
                    if resp.status_code == 200:
                        res_json = resp.json()
                        on_object_detections_received(
                            bounding_boxes=res_json.get("bounding_boxes", []),
                            latency_ms=total_latency_ms
                        )
                    else:
                        print(f"⚠️ API Error ({resp.status_code}): {resp.text[:100]}")

                elif args.mode == "completion":
                    endpoint = f"{base_url}/api/v1/completions"
                    data = {"prompt": args.prompt}
                    if args.model:
                        data["model"] = args.model
                        
                    resp = requests.post(endpoint, files=files, data=data, timeout=10.0)
                    total_latency_ms = (time.perf_counter() - send_start) * 1000.0
                    
                    if resp.status_code == 200:
                        res_json = resp.json()
                        choices = res_json.get("choices", [{}])
                        text_out = choices[0].get("text", "") if choices else ""
                        on_ai_completion_received(
                            text=text_out,
                            model_used=res_json.get("model", str(args.model or "default")),
                            latency_ms=total_latency_ms
                        )
                    else:
                        print(f"⚠️ API Error ({resp.status_code}): {resp.text[:100]}")

            except requests.exceptions.Timeout:
                print(f"⏱️ Network Timeout sending {payload_kb:.1f}KB frame over WAN.")
            except requests.exceptions.RequestException as req_err:
                print(f"🔌 Network Error: {req_err}")

            # Regulate loop to match target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Control loop stopped by user.")
    finally:
        cap.release()
        print("📷 Camera released. Robot client shut down safely.")


if __name__ == "__main__":
    args = parse_args()
    run_client_loop(args)
