#!/usr/bin/env python3
"""
Simple connection and signal test script for Raspberry Pi.
Usage: python test_signal.py
"""

import requests
import cv2
import numpy as np
import time

from discovery import auto_discover_server

SERVER_URL = auto_discover_server()
if not SERVER_URL:
    SERVER_URL = "http://127.0.0.1:8000"

def test_connection():
    print(f"=== 🔍 Testing Connection to {SERVER_URL} ===")
    
    # 1. Test /health endpoint
    print("\n1. Pinging /health endpoint...")
    try:
        start_time = time.time()
        res = requests.get(f"{SERVER_URL}/health", timeout=3.0)
        latency = (time.time() - start_time) * 1000
        
        if res.status_code == 200:
            data = res.json()
            print(f"✅ SUCCESS! Connected in {latency:.1f}ms")
            print(f"💻 GPU Available: {data.get('gpu_available')}")
            print(f"🧠 Loaded Models: {data.get('loaded_models')}")
        else:
            print(f"❌ FAILED. Status Code: {res.status_code}")
            return
    except Exception as e:
        print(f"❌ FAILED. Could not reach {SERVER_URL}/health")
        print(f"Error: {e}")
        return

    # 2. Test sending an image payload
    print("\n2. Generating a dummy test image to send to the server...")
    try:
        # Create a simple 640x480 blue square image using numpy
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image[:] = (255, 0, 0)  # Fill with blue (BGR)
        
        # Compress it to JPEG
        success, encoded_image = cv2.imencode('.jpg', dummy_image)
        if not success:
            print("❌ Failed to encode dummy image.")
            return
            
        image_bytes = encoded_image.tobytes()
        print(f"📦 Payload size: {len(image_bytes)/1024:.1f} KB")
        
        # We will use the chat completions endpoint
        endpoint = f"{SERVER_URL}/v1/chat/completions"
        print(f"🚀 Sending image to {endpoint}...")
        
        start_time = time.time()
        files = {"file": ("test.jpg", image_bytes, "image/jpeg")}
        data = {"prompt": "What color is this image?"}
        res = requests.post(endpoint, files=files, data=data, timeout=10.0)
        latency = (time.time() - start_time) * 1000
        
        if res.status_code == 200:
            data = res.json()
            print(f"✅ SUCCESS! Server processed the image in {latency:.1f}ms")
            print(f"📊 Inference Time (on server): {data.get('inference_time_ms')}ms")
            choices = data.get("choices", [])
            if choices:
                print(f"💬 AI Response: {choices[0].get('message', {}).get('content', choices[0].get('text', ''))}")
            print("\n🎉 The connection is fully working!")
        else:
            print(f"❌ FAILED. Status Code: {res.status_code}")
            print(f"Response: {res.text}")
            
    except Exception as e:
        print(f"❌ FAILED to send image.")
        print(f"Error: {e}")


if __name__ == "__main__":
    test_connection()
