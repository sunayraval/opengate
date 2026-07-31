import time
import subprocess
import requests
import base64
import json
import os

SERVER_URL = "http://172.20.10.2:8000/v1/chat/completions"
IMAGE_PATH = "capture.jpg"

def take_picture():
    print("Taking picture in 3 seconds...")
    time.sleep(3)
    
    # Modern Raspberry Pi OS uses libcamera. 
    # If on legacy OS, you might need to change 'libcamera-jpeg' to 'raspistill -o'
    print("Capturing image...")
    try:
        subprocess.run(["libcamera-jpeg", "-o", IMAGE_PATH, "--nopreview", "-t", "1"], check=True)
        print("Image captured successfully!")
    except FileNotFoundError:
        print("libcamera-jpeg not found. Trying legacy raspistill...")
        subprocess.run(["raspistill", "-o", IMAGE_PATH, "-n", "-t", "1000"], check=True)

def send_request():
    print(f"Sending request to {SERVER_URL}...")
    
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    payload = {
        "model": "openbmb/MiniCPM-V", # The server ignores this and uses the default if not matching exactly, but it's good practice.
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Please describe it."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    try:
        response = requests.post(SERVER_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        description = result['choices'][0]['message']['content']
        elapsed = time.time() - start_time
        
        print("\n" + "="*50)
        print(f"AI RESPONSE (took {elapsed:.2f}s):")
        print("="*50)
        print(description)
        print("="*50 + "\n")
        
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
    finally:
        # Cleanup
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

if __name__ == "__main__":
    take_picture()
    send_request()
