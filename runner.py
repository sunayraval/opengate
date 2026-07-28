import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Runner Dashboard Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
server_process = None
log_clients = []
dashboard_dir = Path(__file__).parent / "dashboard"
loop = None

# Ensure dashboard dir exists
dashboard_dir.mkdir(parents=True, exist_ok=True)

# Mount static files for css/js
app.mount("/static", StaticFiles(directory=str(dashboard_dir)), name="static")

@app.on_event("startup")
async def startup_event():
    global loop
    loop = asyncio.get_running_loop()

def broadcast_log(line: str):
    global loop
    if not loop:
        return
    for client in log_clients:
        try:
            asyncio.run_coroutine_threadsafe(client.send_text(line), loop)
        except Exception:
            pass

def monitor_output(stream):
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            broadcast_log(line)
    except Exception as e:
        pass

@app.post("/api/start")
async def start_server():
    global server_process
    if server_process and server_process.poll() is None:
        return JSONResponse({"status": "already_running"})
    
    env = os.environ.copy()
    # Force unbuffered output so we get logs immediately
    env["PYTHONUNBUFFERED"] = "1"
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(Path(__file__).parent)
    )
    
    t = threading.Thread(target=monitor_output, args=(server_process.stdout,), daemon=True)
    t.start()
    
    return JSONResponse({"status": "started"})

@app.post("/api/stop")
async def stop_server():
    global server_process
    if server_process and server_process.poll() is None:
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        server_process = None
        broadcast_log("[Runner] Server stopped cleanly.\n")
        return JSONResponse({"status": "stopped"})
    return JSONResponse({"status": "already_stopped"})

@app.get("/api/status")
async def get_status():
    global server_process
    is_running = server_process is not None and server_process.poll() is None
    return JSONResponse({"running": is_running})

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    log_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        log_clients.remove(websocket)
    except Exception:
        if websocket in log_clients:
            log_clients.remove(websocket)

@app.get("/")
async def serve_index():
    return FileResponse(str(dashboard_dir / "index.html"))

if __name__ == "__main__":
    print("Starting Runner Dashboard on http://localhost:8080")
    uvicorn.run("runner:app", host="127.0.0.1", port=8080, reload=True)
