#!/usr/bin/env python3
import time
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import os
import sys
import socket
import concurrent.futures

try:
    import serial
except ImportError:
    print("❌ Error: pyserial is not installed. Please run: pip install pyserial")
    sys.exit(1)

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
                
    print("❌ Could not auto-discover server. Defaulting to 127.0.0.1")
    return f"http://127.0.0.1:{port}"

SERVER_URL = auto_discover_server()

def take_picture():
    print("📷 Taking picture...")
    camera_cmds = [
        ["rpicam-jpeg", "-o", IMAGE_PATH, "--width", "320", "--height", "240", "-q", "50", "-t", "100", "--nopreview"],
        ["libcamera-jpeg", "-o", IMAGE_PATH, "--width", "320", "--height", "240", "-q", "50", "-t", "100", "--nopreview"],
        ["raspistill", "-o", IMAGE_PATH, "-w", "320", "-h", "240", "-q", "50", "-t", "100", "-n"]
    ]
    
    success = False
    for cmd in camera_cmds:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            success = True
            break
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            pass
            
    if not success:
        print("⚠️ Camera not found or failed. Falling back to a dummy image for testing.")
        # Create a dummy image if camera fails (so it works on PC)
        from PIL import Image
        img = Image.new('RGB', (320, 240), color = 'gray')
        img.save(IMAGE_PATH)

def send_action(command):
    print(f"🚀 Sending command '{command}' to AI Server...")
    
    import mimetypes
    import uuid
    
    boundary = uuid.uuid4().hex
    headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
    
    # Build multipart payload
    payload = b''
    
    # Add command field
    payload += f'--{boundary}\r\n'.encode('utf-8')
    payload += f'Content-Disposition: form-data; name="command"\r\n\r\n'.encode('utf-8')
    payload += f'{command}\r\n'.encode('utf-8')
    
    # Add image field
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()
        
    payload += f'--{boundary}\r\n'.encode('utf-8')
    payload += f'Content-Disposition: form-data; name="image_file"; filename="capture.jpg"\r\n'.encode('utf-8')
    payload += f'Content-Type: image/jpeg\r\n\r\n'.encode('utf-8')
    payload += image_bytes
    payload += f'\r\n--{boundary}--\r\n'.encode('utf-8')
    
    req = urllib.request.Request(f"{SERVER_URL}/api/v1/communicate/action", data=payload, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30.0) as response:
            print("✅ Server successfully processed the action!")
    except urllib.error.URLError as e:
        print(f"❌ Network Error communicating with server: {e}")
    finally:
        if os.path.exists(IMAGE_PATH):
            os.remove(IMAGE_PATH)

def setup_serial():
    ports_to_try = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1']
    for port in ports_to_try:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"✅ Connected to Arduino on {port}")
            print("⏳ Waiting for Arduino to boot...")
            
            # Wait up to 5 seconds for the Arduino to send its ready message
            start_time = time.time()
            while time.time() - start_time < 5:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "ready" in line.lower():
                    print("✅ Arduino is ready!")
                    return ser
            
            print("⚠️ Did not see ready message, but proceeding anyway.")
            return ser
        except serial.SerialException:
            pass
            
    print("⚠️ Could not connect to Arduino. Actions will not be sent over USB.")
    return None

def main():
    import json
    ser = setup_serial()
    print(f"📡 Pi Continuous Client started. Polling {SERVER_URL} for commands and actions...")
    
    while True:
        # 1. Poll for dashboard commands
        try:
            req = urllib.request.Request(f"{SERVER_URL}/api/v1/commands/pop", method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode('utf-8'))
                command = data.get("command")
                
                if command:
                    print(f"🔔 Received command: {command}")
                    take_picture()
                    send_action(command)
        except Exception:
            pass
            
        # 2. Poll for robot actions
        if ser:
            try:
                req = urllib.request.Request(f"{SERVER_URL}/api/v1/robot/actions/pop", method="GET")
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    actions = data.get("actions", [])
                    
                    if actions:
                        print(f"📥 Received new actions: {actions}")
                        for act in actions:
                            direction = act.get("Direction", "None").upper()
                            magnitude = str(act.get("Magnitude", "0"))
                            
                            if direction != "NONE" and magnitude != "0":
                                command = f"{direction} {magnitude}\n"
                                print(f"🚀 Sending to Arduino: {command.strip()}")
                                ser.write(command.encode('utf-8'))
                                ser.flush()
                                time.sleep(1)
            except Exception:
                pass
                
        time.sleep(1)

if __name__ == "__main__":
    main()
