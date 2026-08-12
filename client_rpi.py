#!/usr/bin/env python3
import time
import subprocess
import urllib.request
import urllib.error
import json
import os
import sys
import socket
import concurrent.futures

try:
    import serial
except ImportError:
    print("❌ Error: pyserial is not installed. Please run: pip install pyserial")
    sys.exit(1)

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

SERVER_URL = auto_discover_server()

def setup_serial():
    ports_to_try = ['/dev/ttyACM0', '/dev/ttyUSB0', '/dev/ttyACM1']
    for port in ports_to_try:
        try:
            ser = serial.Serial(port, 115200, timeout=1)
            print(f"✅ Connected to Arduino on {port}")
            time.sleep(2) # Wait for Arduino to reset
            return ser
        except serial.SerialException:
            pass
            
    print("❌ Error: Could not connect to Arduino. Are you sure it's plugged in via USB?")
    sys.exit(1)

def poll_actions(ser):
    print("🤖 Polling server for robot actions...")
    url = f"{SERVER_URL}/api/v1/robot/actions/pop"
    
    while True:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
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
                                # Wait a bit for Arduino to process before sending the next one
                                time.sleep(1)
                                
        except urllib.error.URLError:
            pass # Server might be down or busy, just retry
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error polling: {e}")
            
        time.sleep(1) # Poll every second

if __name__ == "__main__":
    ser = setup_serial()
    poll_actions(ser)
