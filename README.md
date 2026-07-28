# 🤖 Real-Time AI Computer Vision Robot Framework
**High-Performance Multimodal Autonomy for Raspberry Pi 5 Powered by Windows Desktop GPU & PyTorch VRAM**

Welcome to the **Real-Time AI CV Robot Framework**! This project solves the biggest challenge in hardware robotics: **running massive state-of-the-art Vision-Language Models (MiniCPM-V, OpenCLIP, YOLOv8/11) in real-time on a lightweight Raspberry Pi 5**. 

By hosting the heavy PyTorch AI engine on your **Windows Desktop GPU** and connecting your **Raspberry Pi 5** over your local network (Wi-Fi or LAN), your robot achieves **sub-50ms inference latency**!

> **Note on Remote Access:** If you want your robot to work remotely over cellular networks without port forwarding, read our [cloudflare_setup.md](file:///c:/Users/Immer/Desktop/freeapi-v1.2.3/cloudflare_setup.md) guide!

> [!TIP]
> **Looking for comprehensive API Reference and Multi-Agent tutorials?** Check out our dedicated [docs.md](file:///c:/Users/Immer/Desktop/freeapi-v1.2.3/docs.md) manual for complete examples using Python `requests`, `cURL`, Base64, Image URL Links, and **CrewAI** multi-agent orchestration!

---

## 📑 Table of Contents
1. [🌟 Why This Framework is Built for Autonomy](#-why-this-framework-is-built-for-autonomy)
2. [📁 Repository Anatomy & File Guide](#-repository-anatomy--file-guide)
3. [🖥️ Phase 1: Host Setup (Windows Desktop PC)](#-phase-1-host-setup-windows-desktop-pc)
4. [🌐 Phase 2: Finding Your Local PC IP Address](#-phase-2-finding-your-local-pc-ip-address)
5. [🍓 Phase 3: Raspberry Pi 5 Client Deployment & Execution](#-phase-3-raspberry-pi-5-client-deployment--execution)
6. [💬 Phase 4: Multi-Model AI Usage & Vision-Language Completions](#-phase-4-multi-model-ai-usage--vision-language-completions)
7. [📊 Phase 5: API Endpoints Reference](#-phase-5-api-endpoints-reference)
8. [🧪 Phase 6: Troubleshooting & Diagnostics](#-phase-6-troubleshooting--diagnostics)

---

## 🌟 Why This Framework is Built for Autonomy
* 🔒 **Local Network Privacy**: Everything runs locally over your home Wi-Fi (`http://127.0.0.1:8000` or `http://192.168.x.x:8000`). No cloud subscriptions, no internet connection required for inference.
* ⚡ **Multi-Model GPU Engine**: Keeps multiple vision models warm in PyTorch VRAM simultaneously using half-precision (FP16/BF16) quantization for **2x faster inference** and **50% less VRAM consumption**.
* 📦 **Automatic Binary & Weight Management**: The server automatically downloads `cloudflared.exe` from official releases if missing, and dynamically fetches Hugging Face / YOLO weights on demand.
* 🚀 **Low-Bandwidth WAN Streaming**: The Raspberry Pi compresses camera frames to lightweight JPEG buffers (~25KB) before transmission, dropping network latency over the internet to **< 20ms**.

---

## 📁 Repository Anatomy & File Guide

Here is a complete roadmap of every file and folder in this repository so you know exactly how the system operates:

```text
freeapi-v1.2.3/
├── app/                        # 🧠 PC-Side FastAPI Backend & PyTorch GPU Engine
│   ├── __init__.py             # Package initializer
│   ├── config.py               # Singleton configuration manager (loads .env & hardware settings)
│   ├── main.py                 # Core FastAPI routing, endpoints, and application lifecycle
│   ├── schemas.py              # Pydantic data validation models for API inputs/outputs
│   ├── tunnel.py               # Automated Cloudflare Tunnel manager (downloads & runs cloudflared)
│   └── models/                 # VRAM Model Management Subpackage
│       ├── __init__.py         # Model registry export
│       ├── base.py             # Base abstract class and Multi-Model VRAM LRU Cache Registry
│       ├── clip_model.py       # OpenCLIP Zero-Shot action prediction implementation
│       ├── yolo_model.py       # Ultralytics YOLOv8/11 real-time bounding box detector
│       └── minicpm_model.py    # MiniCPM-V multimodal Vision-Language VQA completion engine
├── client_rpi/                 # 🍓 Lightweight Raspberry Pi Client Codebase
│   ├── requirements_rpi.txt    # Minimal Pi dependencies (OpenCV + Requests + WebSockets)
│   └── robot_client.py         # Autonomous robot client script (captures camera & queries PC)
├── scripts/                    # 🛠️ Developer Utility & Automated Setup Scripts
│   ├── deploy_to_pi.py         # One-command SCP automated deployment tool to transfer files to Pi
│   ├── setup_cloudflare.py     # Interactive wizard & CLI utility for 24/7 permanent URL routing
│   ├── download_models.py      # Pre-download utility to cache AI models offline before deployment
│   └── test_inference.py       # PC local diagnostic script to benchmark GPU latency & VRAM load
├── weights/                    # 💾 Local offline storage directory for downloaded AI models (.pt)
├── docs.md                     # 📚 Comprehensive API reference, Base64/Link examples & CrewAI guide
├── requirements.txt            # 🖥️ Full PC-side Python dependencies (PyTorch, FastAPI, Ultralytics)
└── README.md                   # 📖 This step-by-step master documentation guide!
```

---

## 🖥️ Phase 1: Host Setup (Windows Desktop PC)

Your Windows PC acts as the high-powered AI brain. Follow these 3 simple steps to start your server:

### Step 1: Install Python Dependencies
Open **PowerShell** in this project folder (`freeapi-v1.2.3`) and create a Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Pre-Download AI Models (Offline Cache)
To ensure your server launches instantly without waiting for downloads during robot operation, pre-cache the AI vision models:
```powershell
# Download all supported models (OpenCLIP, YOLOv8n, MiniCPM-V):
python scripts/download_models.py --model all

# Or download a specific model:
python scripts/download_models.py --model minicpm
```

### Step 3: Start the Real-Time AI Server!
Launch the FastAPI GPU backend using the new Web Dashboard. We recommend using `uv` to ensure it automatically finds your virtual environment:
```powershell
uv run runner.py
```
*(If you don't have `uv` configured, you can run `.\venv\Scripts\python.exe runner.py`)*

Open your browser to `http://localhost:8080` to access the **Runner Dashboard**. From here you can start/stop the server, view real-time logs, and monitor API calls!
🎉 **That's it!** The server will start on port `8000`. You can access the **API** locally at `http://127.0.0.1:8000`.

---

## 🌐 Phase 2: Finding Your Local PC IP Address

To connect your Raspberry Pi to your PC, you need to know your Windows Desktop's local IP address on your Wi-Fi or LAN (e.g., `192.168.1.100`).

1. Open a new **PowerShell** window.
2. Type `ipconfig` and press Enter.
3. Look for the `IPv4 Address` under your active network adapter (Wi-Fi or Ethernet).

You will use this IP address when starting the Raspberry Pi client (e.g., `http://192.168.1.100:8000`).

> [!TIP]
> **Want to run this over Cellular / 4G / 5G instead of Local Wi-Fi?**
> See our [cloudflare_setup.md](file:///c:/Users/Immer/Desktop/freeapi-v1.2.3/cloudflare_setup.md) guide to expose your PC securely without port forwarding.

---

## 🍓 Phase 3: Raspberry Pi 5 Client Deployment & Execution

Your Raspberry Pi 5 does **not** need heavy PyTorch or machine learning libraries installed! It only needs OpenCV for capturing camera frames and `requests` for networking.

### Step 1: Transfer Client Files to Your Pi (4 Easy Ways)
Before running anything on the Pi, copy the lightweight `client_rpi/` folder from your PC to your Pi using your preferred method:

* **Method A: Automated Windows Deployment Tool (Recommended)**
  Run our automated SCP script on your PC terminal (requires Wi-Fi/LAN connection and Pi SSH enabled):
  ```powershell
  # Interactive prompt:
  python scripts/deploy_to_pi.py
  # Or direct command:
  python scripts/deploy_to_pi.py --host raspberrypi.local --user pi
  ```
* **Method B: One-Line SCP in PowerShell**
  ```powershell
  scp -r .\client_rpi pi@raspberrypi.local:~/client_rpi
  ```
* **Method C: VS Code / WinSCP Drag-and-Drop**
  Connect to your Pi using VS Code **Remote - SSH** or WinSCP/FileZilla, and drag-and-drop the `client_rpi` folder directly into your Pi's home directory.
* **Method D: Git Clone Directly on the Pi**
  ```bash
  git clone https://github.com/yourusername/freeapi-v1.2.3.git
  cd freeapi-v1.2.3
  ```

### Step 2: Install Minimal Client Dependencies on Pi
Open an SSH terminal on your Raspberry Pi (`ssh pi@raspberrypi.local`) and install the minimal requirements (~30 seconds):
```bash
cd ~/client_rpi
pip install -r requirements_rpi.txt
```

### Step 3: Launch Robot Autonomy!
Connect your USB Webcam or Raspberry Pi Camera Module, and run the client script pointing to your PC's local IP address (replace `192.168.1.100` with your PC's IP):

#### Example 1: Zero-Shot Action Navigation (OpenCLIP)
Ask the AI to evaluate real-time candidate actions based on camera input:
```bash
python robot_client.py --url "http://192.168.1.100:8000" --mode action --actions "move forward,turn left around obstacle,turn right around obstacle,stop immediately"
```
Output arriving in `< 50ms`:
```text
[ 38.2ms] 🎯 DECISION: 'move forward' (Confidence:  92.4%)
[ 41.0ms] 🎯 DECISION: 'turn left around obstacle' (Confidence:  88.1%)
```

#### Example 2: Real-Time Object Detection (YOLOv8 Bounding Boxes)
Detect obstacles, people, and targets at high framerates:
```bash
python robot_client.py --url "http://192.168.1.100:8000" --mode detect --fps 10
```

#### Example 3: Vision-Language Chat / VQA (MiniCPM-V Completions)
Ask complex visual reasoning questions in real-time:
```bash
python robot_client.py --url "http://192.168.1.100:8000" --mode completion --prompt "Describe the pathway ahead and alert me if there is a drop-off or obstacle." --model "openbmb/MiniCPM-V"
```

---

## 💬 Phase 4: Multi-Model AI Usage & Vision-Language Completions

Our server supports standard OpenAI-compatible completion endpoints (`/api/v1/completions` and `/v1/chat/completions`). You can transmit images in **three ways**: via **URL Link (`image_url`)**, via **Base64 JSON**, or via **Multipart Form Upload**.

### 1. Sending Images by URL Link (`image_url`)
Pass any public web or camera link directly in your payload:
```python
import requests

payload = {
    "model": "openbmb/MiniCPM-V",
    "prompt": "Identify any obstacles in this scene.",
    "image_url": "https://images.unsplash.com/photo-1542282088-72c9c27ed0cd"
}
response = requests.post("http://192.168.1.100:8000/api/v1/completions", json=payload)
print("AI Reasoning:", response.json()["choices"][0]["text"])
```

### 2. Sending Images by Base64 JSON
```python
import base64, requests

with open("frame.jpg", "rb") as f:
    b64_str = base64.b64encode(f.read()).decode("utf-8")

payload = {"model": "openbmb/MiniCPM-V", "prompt": "Is the path clear?", "image_base64": b64_str}
response = requests.post("http://192.168.1.100:8000/api/v1/completions", json=payload)
```

### 3. Client Hook for Custom Robotics Logic
In `client_rpi/robot_client.py`, every AI response triggers the real-time callback hook where you can attach custom motor controllers, speech synthesis (TTS), or logging:
```python
def on_ai_completion_received(text: str, model_used: str, latency_ms: float):
    """
    HOOK: Process multimodal Vision-Language completion and VQA responses.
    Attach your motor actuators, speakers, or decision handlers here!
    """
    print(f"[{latency_ms:5.1f}ms] 💬 AI COMPLETION ({model_used}):")
    print(f"   \"{text}\"")
```

### 4. Dynamically Loading New AI Models
Our Multi-Model VRAM Registry allows you to load any Hugging Face or Ultralytics model dynamically per request:
* **New Hugging Face CLIP Models**: Download via `python scripts/download_models.py --model "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"`, then pass `--model "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"` from your Pi client!
* **Custom YOLO Weights**: Drop your trained `.pt` file (e.g., `my_robot.pt`) into the `./weights/` folder on your PC. Then call `--model "my_robot.pt"` from the Pi!
* **MiniCPM-V Vision-Language**: Simply pass `--model "openbmb/MiniCPM-V"` in any action or completion request!

---

## 📊 Phase 5: API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Hardware GPU diagnostics, CUDA VRAM usage (MB), and loaded models list. |
| `GET` | `/api/v1/models` | List all registered vision models and their VRAM load status. |
| `POST` | `/api/v1/completions`<br>`/v1/chat/completions` | Multimodal Vision-Language VQA & chat completions (supports `image_url`, Base64, and multipart files). |
| `POST` | `/api/v1/infer/action` | Zero-shot multimodal action classification (evaluates candidate action strings). |
| `POST` | `/api/v1/infer/detect` | Real-time object detection bounding boxes (`xyxy`), confidences, and class names. |
| `WS` | `/api/v1/stream` | Bidirectional real-time WebSocket video stream for continuous high-speed robot control loops. |

👉 **For advanced API examples, cURL commands, and Multi-Agent CrewAI orchestration tutorials, read our complete [docs.md](file:///c:/Users/Immer/Desktop/freeapi-v1.2.3/docs.md) manual!**

---

## 🧪 Phase 6: Troubleshooting & Diagnostics

### 1. Local PC Benchmark & Test Suite
Before deploying your Pi, verify your GPU VRAM quantization, CUDA acceleration, and inference latency directly on your PC:
```powershell
python scripts/test_inference.py
```
This script generates synthetic test images, runs OpenCLIP and YOLO simultaneously in VRAM, and prints diagnostic timing benchmarks.

### 2. Network Diagnostic Utility
If you ever experience connection issues, ensure your Windows Firewall is not blocking Python, and verify your PC's IP address using `ipconfig`.

### 3. Client API Test Script
To quickly test if your Raspberry Pi can talk to your PC backend before running the full robot autonomous script, we've included a simple request tester:
```bash
python client_rpi/test_request.py
```
This script will ping the `/health` endpoint and send a dummy image to the AI `/infer/action` endpoint to verify the network is working perfectly.

### 4. Server Health & GPU Memory Check
You can query the `/health` endpoint from your browser or terminal anytime to inspect real-time GPU VRAM consumption:
```bash
curl http://127.0.0.1:8000/health
```
Example JSON Response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "device_name": "NVIDIA GeForce RTX 4080",
  "vram_allocated_mb": 4210.5,
  "loaded_models": ["clip-vit-base-patch32", "yolov8n.pt", "openbmb/MiniCPM-V"]
}
```

---
🎉 **Happy Building! You are now ready to deploy state-of-the-art AI vision autonomy to your robotics projects!**
