#!/usr/bin/env python3
import time
import subprocess
import urllib.request
import urllib.error
import base64
import json
import os
import sys
import socket
import concurrent.futures

IMAGE_PATH = "capture.jpg"

def auto_discover_server(port=8000):
    print("🔍 Auto-discovering OpenGate server on the local network...")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        my_ip = s.getsockname()[0]
    except Exception:
        my_ip = '127.0.0.1'
    finally:
        s.close()
        
    base_ip = ".".join(my_ip.split(".")[:-1])
    
    def check_ip(ip):
        url = f"http://{ip}:{port}"
        try:
            req = urllib.request.Request(f"{url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=0.5) as response:
                if response.status == 200:
                    return url
        except Exception:
            pass
        return None

    ips_to_check = [f"{base_ip}.{i}" for i in range(1, 255)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(check_ip, ips_to_check)
        for res in results:
            if res:
                print(f"🎯 Found server automatically at: {res}")
                return res
                
    print("❌ Could not auto-discover server. Defaulting to 172.20.10.2.")
    return "http://172.20.10.2:8000"

SERVER_URL = f"{auto_discover_server()}/v1/chat/completions"

def take_picture():
    print("📷 Initializing Raspberry Pi camera...")
    print("Taking picture in 3 seconds...")
    time.sleep(3)
    
    # Try the newer 'rpicam-jpeg' (Bookworm), then 'libcamera-jpeg' (Bullseye), then 'raspistill' (Buster)
    camera_cmds = [
        ["rpicam-jpeg", "-o", IMAGE_PATH, "--width", "320", "--height", "240", "-q", "50", "-t", "100", "--nopreview"],
        ["libcamera-jpeg", "-o", IMAGE_PATH, "--width", "320", "--height", "240", "-q", "50", "-t", "100", "--nopreview"],
        ["raspistill", "-o", IMAGE_PATH, "-w", "320", "-h", "240", "-q", "50", "-t", "100", "-n"]
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
    if not os.path.exists(IMAGE_PATH):
        print("❌ Error: capture.jpg was not created.")
        sys.exit(1)

def send_request():
    print(f"🚀 Sending request to AI Server at {SERVER_URL}...")
    
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # We use standard JSON to avoid multipart/form-data dependency on the 'requests' library
    payload = {
        "model": "openbmb/MiniCPM-V",
        "prompt": "What do you see in this image? Please describe it.",
        "image_base64": base64_image,
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(SERVER_URL, data=data, headers={"Content-Type": "application/json"})
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30.0) as response:
            result_json = json.loads(response.read().decode('utf-8'))
            
            ai_message = result_json["choices"][0]["message"]["content"]
            elapsed = time.time() - start_time
            
            print("\n" + "="*50)
            print(f"💬 AI ANALYSIS RESULT (took {elapsed:.2f}s):")
            print("="*50)
            print(ai_message)
            print("="*50 + "\n")
            
    except urllib.error.URLError as e:
        print(f"❌ Network Error communicating with server: {e}")
    except json.JSONDecodeError:
        print(f"❌ Error: Server returned an invalid JSON response.")
    finally:
        # Clean up the temporary image file
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

if __name__ == "__main__":
    take_picture()
    send_request()
