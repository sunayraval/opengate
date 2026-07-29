#!/usr/bin/env python3
import requests
import time
import sys
import subprocess
import os
from discovery import auto_discover_server

def capture_and_analyze(prompt="You are a robot. Based on this image, output exactly ONE word to avoid obstacles: 'FORWARD', 'LEFT', 'RIGHT', or 'STOP'."):
    # 1. Auto-discover the OpenGate server
    SERVER_URL = auto_discover_server()
    if not SERVER_URL:
        print("⚠️ Could not find server automatically. Defaulting to localhost.")
        SERVER_URL = "http://127.0.0.1:8000"
        
    endpoint = f"{SERVER_URL}/v1/chat/completions"
    
    # 2. Capture using native Raspberry Pi camera commands
    print("📷 Initializing Raspberry Pi camera...")
    
    # Try the newer 'rpicam-jpeg' (Bookworm), then 'libcamera-jpeg' (Bullseye), then 'raspistill' (Buster)
    camera_cmds = [
        ["rpicam-jpeg", "-o", "capture.jpg", "--width", "320", "--height", "240", "-q", "50", "-t", "100", "--nopreview"],
        ["libcamera-jpeg", "-o", "capture.jpg", "--width", "320", "--height", "240", "-q", "50", "-t", "100", "--nopreview"],
        ["raspistill", "-o", "capture.jpg", "-w", "320", "-h", "240", "-q", "50", "-t", "100", "-n"]
    ]
    
    print("📸 Snapping a picture...")
    success = False
    for cmd in camera_cmds:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            success = True
            break  # It worked! Break out of the loop.
        except FileNotFoundError:
            continue  # Command not found on this OS version, try the next one
        except subprocess.CalledProcessError:
            print(f"❌ Error: {cmd[0]} ran but failed to capture an image. Check your camera ribbon cable!")
            sys.exit(1)
            
    if not success:
        print("❌ Error: Could not find rpicam-jpeg, libcamera-jpeg, or raspistill on this system. Is the camera enabled?")
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
        "prompt": prompt,
        "max_tokens": "5"
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
    user_prompt = "You are a robot. Based on this image, output exactly ONE word to avoid obstacles: 'FORWARD', 'LEFT', 'RIGHT', or 'STOP'."
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        
    capture_and_analyze(prompt=user_prompt)
