# OpenGate API Server

OpenGate is a lightweight, local API provider for Vision-Language Models (VLMs) and LLMs. It exposes an OpenAI-compatible API endpoint, allowing you to seamlessly integrate local models into your scripts, web applications, or tools. 

It handles everything from routing and VRAM management to image decoding, making local AI inference as easy as calling a cloud provider.

---

## 📑 Table of Contents
1. [Getting Started](#getting-started)
2. [Starting the Server](#starting-the-server)
3. [API Reference](#api-reference)
   - [Endpoints](#endpoints)
4. [Payload Structures & Examples](#payload-structures--examples)
   - [1. Text-Only Completions](#1-text-only-completions)
   - [2. Vision: Base64 Encoded Images (JSON)](#2-vision-base64-encoded-images-json)
   - [3. Vision: HTTP Image Links (JSON)](#3-vision-http-image-links-json)
   - [4. Vision: File Attachments (Multipart Form)](#4-vision-file-attachments-multipart-form)
5. [Parsing the Response](#parsing-the-response)

---

## Getting Started

### Prerequisites
1. Python 3.10+
2. A CUDA-capable NVIDIA GPU (recommended for speed) or CPU.

### Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Starting the Server

To launch the OpenGate API server on your local machine, run:
```bash
python runner.py
```
*(Alternatively, you can run `python app/main.py` directly).*

By default, the API will be available at:
`http://127.0.0.1:8000`

---

## API Reference

### Endpoints
- **GET `/health`**
  Returns system diagnostics, GPU hardware status, VRAM allocation, and currently loaded models.
- **GET `/api/v1/models`**
  Lists all available models in the local registry.
- **POST `/v1/chat/completions`**
  *(Aliases: `/api/v1/completions`, `/completions`)*
  The primary endpoint for sending text and images to generate AI completions.

---

## Payload Structures & Examples

The `/v1/chat/completions` endpoint is highly flexible and accepts multiple payload formats to match standard API specifications (like OpenAI).

### 1. Text-Only Completions
Send a simple JSON payload with an array of messages or a direct prompt.

```python
import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "openbmb/MiniCPM-V",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
}

response = requests.post(url, json=payload)
print(response.json())
```

### 2. Vision: Base64 Encoded Images (JSON)
You can embed an image directly in the JSON payload using Base64 encoding.

```python
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("path/to/image.jpg")

url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "openbmb/MiniCPM-V",
    "prompt": "Describe this image in detail.",
    "image_base64": base64_image,
    "max_tokens": 512
}

response = requests.post(url, json=payload)
print(response.json())
```

### 3. Vision: HTTP Image Links (JSON)
If your image is already hosted online, just pass the URL. The server will download it automatically.

```python
import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "openbmb/MiniCPM-V",
    "prompt": "What color is the car in this picture?",
    "image_url": "https://example.com/path/to/car.jpg"
}

response = requests.post(url, json=payload)
print(response.json())
```

### 4. Vision: File Attachments (Multipart Form)
If you don't want to deal with Base64 strings, you can upload the file directly using `multipart/form-data`. This is highly efficient and easy to use.

```python
import requests

url = "http://127.0.0.1:8000/v1/chat/completions"
files = {
    "file": ("image.jpg", open("path/to/image.jpg", "rb"), "image/jpeg")
}
data = {
    "model": "openbmb/MiniCPM-V",
    "prompt": "Read the text in this image."
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

---

## Parsing the Response

The response strictly follows the standard OpenAI API structure, making it drop-in compatible with many existing scripts and libraries.

**Example Response Body:**
```json
{
  "id": "cmpl-1718049182300",
  "object": "chat.completion",
  "created": 1718049182,
  "model": "openbmb/MiniCPM-V",
  "choices": [
    {
      "index": 0,
      "text": "The image shows a red sports car parked in front of a modern building.",
      "message": {
        "role": "assistant",
        "content": "The image shows a red sports car parked in front of a modern building."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 15,
    "total_tokens": 29
  },
  "inference_time_ms": 1250.4,
  "raw_response": "The image shows a red sports car parked in front of a modern building."
}
```

**Python Parsing Example:**
```python
data = response.json()

# Extract the text content
ai_message = data["choices"][0]["message"]["content"]
print("AI Says:", ai_message)

# Check latency
print(f"Generated in {data['inference_time_ms']} ms")
```
