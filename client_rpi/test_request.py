import requests
import json
import base64
import os

# Configuration
# Change this to the IP address of your PC if running on a different machine, or the Cloudflare URL
SERVER_URL = "http://127.0.0.1:8000"

def test_health():
    """Test the basic health endpoint of the API."""
    print(f"Testing connection to {SERVER_URL}/health...")
    try:
        response = requests.get(f"{SERVER_URL}/health")
        response.raise_for_status()
        print("Success! Health Check Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to connect to health endpoint: {e}")

def test_action_inference():
    """Test the AI action inference endpoint by sending a dummy image."""
    print(f"\nTesting action inference at {SERVER_URL}/api/v1/infer/action...")
    
    # Create a 1x1 black pixel image in memory (for testing purposes)
    # In a real scenario, this would be a frame captured from the Raspberry Pi camera
    dummy_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    payload = {
        "image_base64": dummy_image_base64,
        "candidate_actions": ["move forward", "turn left", "turn right", "stop"],
        "model_name": "clip-vit-base-patch32" # Use default model
    }

    try:
        response = requests.post(f"{SERVER_URL}/api/v1/infer/action", json=payload)
        response.raise_for_status()
        print("Success! Inference Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Failed to get inference result: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server replied: {e.response.text}")

if __name__ == "__main__":
    print("--- FreeAPI Client Test Script ---")
    test_health()
    test_action_inference()
