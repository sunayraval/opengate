import socket
import concurrent.futures
import requests

def auto_discover_server(port=8000):
    """
    Scans the local subnet to automatically find the OpenGate server running on the specified port.
    Returns the full base URL string (e.g. 'http://192.168.1.5:8000') if found, or None if not found.
    """
    print("🔍 Auto-discovering OpenGate server on the local network...")
    # Get the Pi's own local IP address
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
            res = requests.get(f"{url}/health", timeout=0.5)
            if res.status_code == 200:
                return url
        except Exception:
            pass
        return None

    # Scan all IPs in the subnet concurrently (takes < 1 second)
    ips_to_check = [f"{base_ip}.{i}" for i in range(1, 255)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(check_ip, ips_to_check)
        for res in results:
            if res:
                print(f"🎯 Found server automatically at: {res}")
                return res
                
    print("❌ Could not auto-discover server. Are you on the same Hotspot?")
    return None
