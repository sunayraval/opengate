#!/usr/bin/env python3
import cv2
import requests
import time
import sys
from discovery import auto_discover_server

def capture_and_analyze(prompt="What do you see in this image? Describe it in detail."):
    # 1. Auto-discover the OpenGate server
    SERVER_URL = auto_discover_server()
    if not SERVER_URL:
        print("⚠️ Could not find server automatically. Defaulting to localhost.")
        SERVER_URL = "http://127.0.0.1:8000"
        
    endpoint = f"{SERVER_URL}/v1/chat/completions"
    
    # 2. Open the Pi Camera (Camera index 0)
    print("📷 Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Optional: Set resolution to 640x480 to keep the upload fast over the hotspot
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not open the camera. Check your Pi camera connection.")
        sys.exit(1)
        
    # Warm up camera for a second so auto-exposure can adjust
    time.sleep(1.0)
    
    print("📸 Snapping a picture...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print("❌ Error: Failed to capture an image from the camera.")
        sys.exit(1)
        
    # 3. Compress the image to JPEG
    success, encoded_image = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        print("❌ Error: Failed to encode the image.")
        sys.exit(1)
        
    image_bytes = encoded_image.tobytes()
    print(f"📦 Captured image! Size: {len(image_bytes)/1024:.1f} KB")
    
    # 4. Send the image to the AI Server
    print(f"🚀 Sending to AI Server at {endpoint}...")
    
    files = {
        "file": ("capture.jpg", image_bytes, "image/jpeg")
    }
    data = {
        "model": "openbmb/MiniCPM-V",
        "prompt": prompt
    }
    
    try:
        start_time = time.time()
        res = requests.post(endpoint, files=files, data=data, timeout=30.0)
        latency = (time.time() - start_time) * 1000
        
        if res.status_code == 200:
            result_json = res.json()
            ai_message = result_json["choices"][0]["message"]["content"]
            print("\n========================================")
            print("💬 AI ANALYSIS RESULT:")
            print("========================================")
            print(ai_message)
            print("========================================")
            print(f"⏱️ Inference time (Total): {latency:.1f} ms")
            print(f"⏱️ Inference time (Server-side): {result_json.get('inference_time_ms', 0)} ms")
        else:
            print(f"❌ Server Error {res.status_code}: {res.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")

if __name__ == "__main__":
    # You can pass a custom prompt as a command line argument!
    user_prompt = "What do you see in this image? Describe it in detail."
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        
    capture_and_analyze(prompt=user_prompt)
