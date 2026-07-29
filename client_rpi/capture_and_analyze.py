#!/usr/bin/env python3
import requests
import time
import sys
import subprocess
import os
from discovery import auto_discover_server

def capture_and_analyze(prompt="What do you see in this image? Describe it in detail."):
    # 1. Auto-discover the OpenGate server
    SERVER_URL = auto_discover_server()
    if not SERVER_URL:
        print("⚠️ Could not find server automatically. Defaulting to localhost.")
        SERVER_URL = "http://127.0.0.1:8000"
        
    endpoint = f"{SERVER_URL}/v1/chat/completions"
    
    # 2. Capture using native Raspberry Pi libcamera command
    print("📷 Initializing Raspberry Pi camera...")
    
    # We use libcamera-jpeg because it's the native, guaranteed way to 
    # take pictures on modern Raspberry Pi OS without fighting OpenCV drivers.
    # -t 1000 gives the camera 1 second to warm up and adjust exposure.
    command = [
        "libcamera-jpeg",
        "-o", "capture.jpg",
        "--width", "640",
        "--height", "480",
        "-t", "1000",
        "--nopreview"
    ]
    
    print("📸 Snapping a picture...")
    try:
        # Run the command and wait for it to finish
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("❌ Error: 'libcamera-jpeg' command not found. Are you running this on a Raspberry Pi with the camera enabled?")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("❌ Error: libcamera failed to capture an image. Check your camera ribbon cable!")
        sys.exit(1)
        
    # Read the captured image from disk
    if not os.path.exists("capture.jpg"):
        print("❌ Error: capture.jpg was not created.")
        sys.exit(1)
        
    with open("capture.jpg", "rb") as f:
        image_bytes = f.read()
        
    print(f"📦 Captured image! Size: {len(image_bytes)/1024:.1f} KB")
    
    # 3. Send the image to the AI Server
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
    finally:
        # Clean up the temporary image file
        if os.path.exists("capture.jpg"):
            os.remove("capture.jpg")

if __name__ == "__main__":
    # You can pass a custom prompt as a command line argument!
    user_prompt = "What do you see in this image? Describe it in detail."
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        
    capture_and_analyze(prompt=user_prompt)
