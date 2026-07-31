# 📚 Real-Time AI Vision & Completion API Documentation

Welcome to the official API documentation for the **Real-Time AI CV Framework**. Our server provides high-performance, low-latency endpoints for Multimodal Vision-Language Question Answering (VQA), Zero-Shot Navigation, and Real-Time Object Detection.

> **Note on Model Selection:** We have replaced custom, brittle models (microsoft/Mage-VL and Qwen/Qwen2-VL-7B-Instruct) with native, robust, state-of-the-art HuggingFace VLMs (Qwen/Qwen2-VL-7B-Instruct and microsoft/Phi-3.5-vision-instruct). This was done for long-term stability: native models don't require 	rust_remote_code=True, meaning they don't download unversioned python scripts from HuggingFace that break whenever the 	ransformers library updates.

---

## 🚀 Endpoint Overview

| Endpoint | Method | Supported Inputs | Primary Use Case |
| :--- | :--- | :--- | :--- |
| `/api/v1/completions`<br>`/v1/chat/completions` | `POST` | JSON (Base64 / URL Link)<br>Multipart Form (File Upload) | Multimodal VQA, scene reasoning, and visual chat (e.g., `Qwen/Qwen2-VL-7B-Instruct`). |
| `/api/v1/infer/action` | `POST` | JSON (Base64 / URL Link)<br>Multipart Form (File Upload) | Zero-shot decision making between candidate actions (e.g., OpenCLIP). |
| `/api/v1/infer/detect` | `POST` | JSON (Base64 / URL Link)<br>Multipart Form (File Upload) | Bounding box object detection and tracking (e.g., YOLOv8). |
| `/api/v1/models` | `GET` | None | Querying currently available and loaded GPU VRAM models. |

---

## 🖼️ Part 1: All Image Transmission Modes (`requests` & `curl`)

Our API is designed for maximum flexibility. You can send images in **three different ways** depending on your network and client setup: by **URL Link**, by **Base64 JSON**, or by **Multipart Form File Upload**.

### 1. Sending Images by Link (`image_url`)
If your image is hosted online (e.g., on S3, Cloudflare R2, or a public camera feed), you can simply pass the HTTP/HTTPS link. The server will download and process it automatically!

#### Python (`requests`) Example:
```python
import requests

url = "http://127.0.0.1:8000/api/v1/completions" # Or your TryCloudflare URL
payload = {
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "prompt": "Identify any potential hazards or obstacles in this path.",
    "image_url": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd",
    "temperature": 0.5,
    "max_tokens": 256
}

response = requests.post(url, json=payload)
data = response.json()
print("AI Completion:", data["choices"][0]["text"])
```

#### cURL Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "Qwen/Qwen2-VL-7B-Instruct",
           "prompt": "What objects are visible in this image?",
           "image_url": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd"
         }'
```

---

### 2. Sending Images by Base64 Encoding (`image_base64`)
Ideal for JSON-only pipelines or when capturing live frames from OpenCV / PiCam where saving to disk is unnecessary.

#### Python (`requests`) Example:
```python
import base64
import requests

# Encode local image file or in-memory frame buffer to Base64
with open("camera_frame.jpg", "rb") as img_file:
    b64_str = base64.b64encode(img_file.read()).decode("utf-8")

payload = {
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "prompt": "Describe the scene and suggest a safe navigation heading.",
    "image_base64": b64_str,
    "temperature": 0.7
}

response = requests.post("http://127.0.0.1:8000/api/v1/completions", json=payload)
print("AI Reasoning:", response.json()["choices"][0]["text"])
```

#### OpenAI Messages Format (Data URI Support):
You can also send requests using the standard OpenAI Multimodal Chat schema:
```python
payload = {
    "model": "Qwen/Qwen2-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Is there a person in front of the robot?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}}
            ]
        }
    ]
}
response = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=payload)
```

---

### 3. Sending Images by Multipart Form Upload
This is the **fastest method for real-time robotics** (~20ms latency) because it avoids the 33% payload bloat of Base64 encoding.

#### Python (`requests`) Example:
```python
import requests

with open("camera_frame.jpg", "rb") as f:
    files = {"file": ("frame.jpg", f, "image/jpeg")}
    data = {
        "prompt": "Are the traffic lights red or green?",
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "temperature": 0.2
    }
    response = requests.post("http://127.0.0.1:8000/api/v1/completions", files=files, data=data)

print("Result:", response.json()["choices"][0]["text"])
```

#### cURL Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/completions" \
     -F "file=@/path/to/camera_frame.jpg" \
     -F "prompt=Analyze this scene for autonomous navigation." \
     -F "model=Qwen/Qwen2-VL-7B-Instruct"
```

---

### 4. Text-Only Conversation & Reasoning
You can also use the `/completions` endpoint without an image for pure LLM reasoning, planning, or text summarization:
```python
response = requests.post(
    "http://127.0.0.1:8000/api/v1/completions",
    json={
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "prompt": "If a robot detects a sudden drop in terrain, what emergency action should it take?",
        "temperature": 0.3
    }
)
print("Response:", response.json()["choices"][0]["text"])
```

---

## 🤖 Part 2: Multi-Agent Orchestration with CrewAI (`crewai`)

You can empower **CrewAI** agents with real-time vision capabilities by integrating our API! There are two primary integration patterns:
1. **Custom Vision Tools (`@tool`)**: Giving agents a dedicated tool to inspect camera frames or online images.
2. **Custom LLM Configuration**: Using our PC's GPU server as the backend LLM for your autonomous crew.

### Pattern 1: Creating a Custom Vision Tool for CrewAI
Using the `@tool` decorator from `crewai.tools`, we can build a reusable vision inspection tool that utilizes the `requests` module.

```python
import base64
import requests
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# Define your Desktop server URL
SERVER_URL = "http://127.0.0.1:8000" # Replace with your TryCloudflare or local URL

@tool("Robot Scene Inspector")
def inspect_scene(image_source: str, prompt: str) -> str:
    """
    Analyzes an image from a web link (http/https), local file path, or Base64 string
    to answer visual questions about the robot's surroundings.
    
    Args:
        image_source: HTTP link (e.g. 'https://...'), local filepath (e.g. 'frame.jpg'), or Base64 string.
        prompt: Question or analysis instruction for the vision model.
    """
    endpoint = f"{SERVER_URL}/api/v1/completions"
    
    # 1. Handle HTTP/HTTPS Links
    if image_source.startswith("http://") or image_source.startswith("https://"):
        payload = {"model": "Qwen/Qwen2-VL-7B-Instruct", "prompt": prompt, "image_url": image_source}
        res = requests.post(endpoint, json=payload, timeout=10.0)
        return res.json().get("choices", [{}])[0].get("text", "Error inspecting scene.")
        
    # 2. Handle Local File Paths
    elif image_source.endswith((".jpg", ".jpeg", ".png")):
        try:
            with open(image_source, "rb") as f:
                files = {"file": ("frame.jpg", f, "image/jpeg")}
                data = {"model": "Qwen/Qwen2-VL-7B-Instruct", "prompt": prompt}
                res = requests.post(endpoint, files=files, data=data, timeout=10.0)
                return res.json().get("choices", [{}])[0].get("text", "Error inspecting file.")
        except Exception as e:
            return f"File error: {e}"
            
    # 3. Handle Raw Base64
    else:
        payload = {"model": "Qwen/Qwen2-VL-7B-Instruct", "prompt": prompt, "image_base64": image_source}
        res = requests.post(endpoint, json=payload, timeout=10.0)
        return res.json().get("choices", [{}])[0].get("text", "Error inspecting base64.")
```

---

### Pattern 2: Complete Autonomous CrewAI Example
Here is a complete, runnable script showing an **Autonomous Reconnaissance Crew** consisting of a **Visual Scout Agent** and a **Navigation Planner Agent** collaborating to navigate a robot safely!

```python
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
import requests

# 1. Define the Vision Tool
SERVER_URL = "http://127.0.0.1:8000"

@tool("Robot Scene Inspector")
def inspect_scene(image_source: str, prompt: str) -> str:
    """
    Analyzes an image from a web link (http/https) or local file path to answer visual questions.
    """
    endpoint = f"{SERVER_URL}/api/v1/completions"
    if image_source.startswith("http://") or image_source.startswith("https://"):
        res = requests.post(endpoint, json={"model": "Qwen/Qwen2-VL-7B-Instruct", "prompt": prompt, "image_url": image_source})
        return res.json()["choices"][0]["text"]
    with open(image_source, "rb") as f:
        res = requests.post(endpoint, files={"file": f}, data={"model": "Qwen/Qwen2-VL-7B-Instruct", "prompt": prompt})
        return res.json()["choices"][0]["text"]

# 2. Define the CrewAI Agents
scout_agent = Agent(
    role="Autonomous Visual Scout",
    goal="Inspect camera frames and accurately identify obstacles, terrain, and hazards.",
    backstory="You are the AI eyes of a tactical reconnaissance rover. Your job is to report objective visual facts.",
    tools=[inspect_scene],
    verbose=True
)

navigator_agent = Agent(
    role="Lead Navigation Planner",
    goal="Formulate safe waypoint routing and action commands based on scout reconnaissance reports.",
    backstory="You are an expert roboticist and pathfinder. You turn visual descriptions into precise navigation commands (move forward, turn left, turn right, stop).",
    verbose=True
)

# 3. Define Tasks
scout_task = Task(
    description="Use the Robot Scene Inspector tool to examine 'https://images.unsplash.com/photo-1542282088-72c9c27ed0cd' and identify all potential obstacles.",
    expected_output="A detailed visual report listing obstacles, pathway clearance, and terrain conditions.",
    agent=scout_agent
)

navigation_task = Task(
    description="Analyze the Visual Scout's report and decide on the single best navigation command (e.g., MOVE FORWARD, STOP, TURN LEFT, TURN RIGHT) with justification.",
    expected_output="Final navigation command and safety justification.",
    agent=navigator_agent
)

# 4. Assemble and Run the Crew
rover_crew = Crew(
    agents=[scout_agent, navigator_agent],
    tasks=[scout_task, navigation_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":
    print("🤖 Launching Autonomous Rover CrewAI Orchestration...")
    result = rover_crew.kickoff()
    print("\n🏁 FINAL NAVIGATION DECISION:\n", result)
```

---

## ⚡ Part 3: Zero-Shot Action & Object Detection Calls

In addition to VQA completions, you can call our specialized inference endpoints with `requests` using any of the image transmission modes:

### Zero-Shot Action Navigation (`/api/v1/infer/action`)
```python
import requests

url = "http://127.0.0.1:8000/api/v1/infer/action"
payload = {
    "model_name": "clip-vit-base-patch32",
    "candidate_actions": ["move forward", "turn left around obstacle", "stop immediately"],
    "image_url": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd"
}

response = requests.post(url, json=payload)
data = response.json()
print(f"🎯 Selected Action: '{data['action']}' (Confidence: {data['confidence']*100:.1f}%)")
```

### Real-Time Object Detection (`/api/v1/infer/detect`)
```python
import requests

url = "http://127.0.0.1:8000/api/v1/infer/detect"
payload = {
    "model_name": "yolov8n.pt",
    "image_url": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd"
}

response = requests.post(url, json=payload)
data = response.json()
print(f"📦 Detected {len(data['bounding_boxes'])} objects:")
for box in data["bounding_boxes"]:
    print(f"   -> {box['class_name']} ({box['confidence']*100:.1f}%)")
```

---



