import subprocess
import requests
import os
import sys

# Replace with the exact IP Address shown in your blue dashboard badge!
SERVER_URL = "http://YOUR_SERVER_IP:8000/v1/chat/completions"
IMAGE_PATH = "gemma_capture.jpg"

def take_picture():
    print("📷 Taking picture with Raspberry Pi Camera...")
    
    # Try modern Raspberry Pi camera commands (Bookworm / Bullseye / Buster)
    # Using 320x240 resolution for ultra-fast AI processing (sub-second latency)
    camera_cmds = [
        ["rpicam-jpeg", "-o", IMAGE_PATH, "--width", "320", "--height", "240", "-q", "80", "-t", "500", "--nopreview"],
        ["libcamera-jpeg", "-o", IMAGE_PATH, "--width", "320", "--height", "240", "-q", "80", "-t", "500", "--nopreview"],
        ["raspistill", "-o", IMAGE_PATH, "-w", "320", "-h", "240", "-q", "80", "-t", "500", "-n"]
    ]
    
    for cmd in camera_cmds:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(IMAGE_PATH):
                return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
            
    print("❌ Error: Could not capture image. Is the camera ribbon cable connected?")
    sys.exit(1)

def send_to_gemma():
    print(f"🚀 Sending image to Gemma-3 at {SERVER_URL}...")
    
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
    
    # We use multipart/form-data via requests for high performance
    files = {
        "file": ("capture.jpg", image_bytes, "image/jpeg")
    }
    data = {
        "model": "google/gemma-3-4b-it",
        "prompt": "You are a fast robot vision system. Describe what you see in under 10 words.",
        "max_tokens": 50,
        "temperature": 0.4
    }
    
    try:
        response = requests.post(SERVER_URL, files=files, data=data)
        if response.status_code == 200:
            # Gemma 3 successfully processed the image!
            ai_message = response.json().get("raw_response", response.text)
            print("\n" + "="*50)
            print("💬 GEMMA 3 SAYS:")
            print("="*50)
            print(ai_message)
            print("="*50 + "\n")
        else:
            print(f"❌ Server Error {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
        print("Did you replace YOUR_SERVER_IP with the IP address from the dashboard?")
    finally:
        # Cleanup the image so we don't fill the SD card
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

if __name__ == "__main__":
    if take_picture():
        send_to_gemma()
