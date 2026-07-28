#!/usr/bin/env python3
"""
Cloudflare Tunnel Setup & Verification CLI Helper Script.

This script guides developers and robot operators through configuring Cloudflare Tunnels
for secure, low-latency communication between the PC-side AI backend and remote robots
(such as Raspberry Pi client over cellular or external Wi-Fi networks).
"""

import argparse
import sys
from pathlib import Path

# Add workspace root to sys.path to allow importing app package
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    from app.tunnel import CloudflareTunnelManager
except ImportError as e:
    print(f"[ERROR] Could not import CloudflareTunnelManager from app.tunnel: {e}", file=sys.stderr)
    print("Ensure you are running this script from the project root or with correct PYTHONPATH.")
    sys.exit(1)


def print_banner():
    banner = """
================================================================================
           REAL-TIME AI CV ROBOT FRAMEWORK - CLOUDFLARE NETWORKING GUIDE
================================================================================

Cloudflare Tunnels (cloudflared) create a secure, encrypted HTTPS/WSS connection
between your local PC (running heavy PyTorch AI models) and your remote robot
(e.g., Raspberry Pi) without needing public static IPs, port forwarding, or VPNs!

There are two primary modes supported by this framework:

--------------------------------------------------------------------------------
MODE 1: QUICK TUNNELS (Dev / Ephemeral Mode - Zero Account Required!)
--------------------------------------------------------------------------------
* Perfect for testing, development, and quick demos.
* When you launch the FastAPI server (python -m uvicorn app.main:app), the framework
  automatically starts an anonymous ephemeral tunnel.
* Cloudflare assigns a random public URL (e.g., https://cool-robot-123.trycloudflare.com).
* No login, credit card, or DNS setup is required!
* Note: The URL changes each time you restart the PC server.

--------------------------------------------------------------------------------
MODE 2: STATIC ZERO TRUST TUNNELS (Production / Persistent Mode)
--------------------------------------------------------------------------------
* Recommended for field deployments where the robot needs a permanent, fixed URL
  (e.g., https://api.myrobotdomain.com and wss://api.myrobotdomain.com/api/v1/stream).
* Setup Steps:
  1. Log into Cloudflare Zero Trust dashboard (https://one.dash.cloudflare.com).
  2. Navigate to Networks -> Tunnels -> Create a Tunnel (Cloudflared).
  3. Name your tunnel (e.g., 'robot-pc-backend') and copy the provided Tunnel Token.
  4. Configure a Public Hostname route pointing to http://localhost:8000.
  5. Add your token to your local .env file or export it as an environment variable:
     
     CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoiZDI5...
     STATIC_DOMAIN=https://api.myrobotdomain.com

--------------------------------------------------------------------------------
ALTERNATIVE: DUCKDNS / CUSTOM DDNS
--------------------------------------------------------------------------------
* If you prefer direct port forwarding with DuckDNS instead of Cloudflare:
  1. Forward TCP port 8000 on your home router to your PC's local IP.
  2. Set STATIC_DOMAIN=http://myrobot.duckdns.org:8000 in your .env file.
================================================================================
"""
    print(banner)


def verify_binary(manager: CloudflareTunnelManager, force_download: bool = False):
    print("\n--- Checking cloudflared binary status ---")
    if force_download:
        print("[INFO] Forcing re-download of official cloudflared executable...")
        try:
            exe_path = manager.download_cloudflared()
            print(f"[SUCCESS] Downloaded binary to: {exe_path}")
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            return
    else:
        try:
            exe_path = manager.find_or_download_cloudflared()
            print(f"[SUCCESS] cloudflared binary ready at: {exe_path}")
        except Exception as e:
            print(f"[ERROR] Could not locate or download cloudflared: {e}")
            return

    # Print version test
    import subprocess
    try:
        res = subprocess.run([str(exe_path), "--version"], capture_output=True, text=True, check=True)
        print(f"[VERSION] {res.stdout.strip() or res.stderr.strip()}")
    except Exception as e:
        print(f"[WARNING] Binary found, but failed to run '--version' check: {e}")


def update_env_file(key: str, value: str):
    """
    Creates or updates a key-value pair in the local .env file.
    """
    env_path = workspace_root / ".env"
    lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
    key_found = False
    new_lines = []
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            key_found = True
        else:
            new_lines.append(line)
            
    if not key_found:
        new_lines.append(f"{key}={value}\n")
        
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"✅ Saved {key} to {env_path}")


def run_interactive_wizard(manager: CloudflareTunnelManager):
    print("\n" + "="*76)
    print("🧙 24/7 PERMANENT STATIC URL SETUP WIZARD")
    print("="*76)
    print("\nTo hardcode a permanent URL into your Raspberry Pi 5, we will configure")
    print("a Cloudflare Zero Trust Tunnel Token or static domain in your local .env file.\n")
    print("Step 1: Get your Free Tunnel Token:")
    print("  -> Go to https://one.dash.cloudflare.com (Zero Trust Dashboard)")
    print("  -> Navigate to: Networks -> Tunnels -> Create a Tunnel (select Cloudflared)")
    print("  -> Name it (e.g. 'robot-backend') and copy the Tunnel Token string.")
    print("  -> Add a Public Hostname route pointing to http://localhost:8000\n")
    
    token = input("🔑 Paste your Cloudflare Tunnel Token (or press Enter to skip): ").strip()
    if token:
        update_env_file("CLOUDFLARE_TUNNEL_TOKEN", token)
        
    domain = input("🌐 Enter your custom public HTTPS domain (e.g., https://api.myrobot.com) [Optional]: ").strip()
    if domain:
        if not domain.startswith("http"):
            domain = f"https://{domain}"
        update_env_file("STATIC_DOMAIN", domain)
        
    if token or domain:
        print("\n🎉 Configuration Saved to .env! Let's test your permanent tunnel now...")
        manager.token = token or manager.token
        manager.static_domain = domain or manager.static_domain
        test_quick_tunnel(manager)
    else:
        print("\n⚠️ No changes made. You can run this wizard anytime or use --token directly!")


def test_quick_tunnel(manager: CloudflareTunnelManager):
    print("\n--- Testing Tunnel Launch ---")
    if manager.token:
        print("Starting Permanent Zero-Trust Tunnel using saved token...")
    else:
        print("Starting temporary ephemeral tunnel forwarding to http://localhost:8000...")
    try:
        url = manager.start(timeout_sec=15)
        if url:
            print(f"\n[SUCCESS] Tunnel successfully established!")
            print(f"Public URL: {url}")
            print(f"WebSocket Stream URL: {url.replace('https://', 'wss://').replace('http://', 'ws://')}/api/v1/stream")
        else:
            print("\n[WARNING] Tunnel started, but public URL was not detected within timeout.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start tunnel: {e}")
    finally:
        print("\nStopping test tunnel process...")
        manager.stop()
        print("Test tunnel closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Cloudflare Tunnel Setup and Diagnostic Utility")
    parser.add_argument("--check", action="store_true", help="Check for cloudflared binary and download if missing.")
    parser.add_argument("--download", action="store_true", help="Force re-download of official cloudflared binary from GitHub.")
    parser.add_argument("--test-quick", action="store_true", help="Test launching a tunnel and print the public URL.")
    parser.add_argument("--info", action="store_true", help="Display comprehensive setup guide and documentation.")
    parser.add_argument("--wizard", action="store_true", help="Interactive setup wizard to save a permanent 24/7 static URL/token.")
    parser.add_argument("--token", type=str, help="Directly save a Cloudflare Tunnel Token to .env for permanent static routing.")
    parser.add_argument("--domain", type=str, help="Directly save a static domain (e.g. https://api.myrobot.com) to .env.")
    parser.add_argument("--duckdns", type=str, help="Directly configure DuckDNS / DDNS port forwarding URL in .env.")
    
    args = parser.parse_args()

    # If no arguments passed, print banner and perform check
    if not any(vars(args).values()):
        print_banner()
        manager = CloudflareTunnelManager()
        verify_binary(manager)
        print("\nTip: Run 'python scripts/setup_cloudflare.py --wizard' to configure your permanent 24/7 static URL!")
        return

    if args.info:
        print_banner()
        return

    manager = CloudflareTunnelManager()

    if args.token:
        update_env_file("CLOUDFLARE_TUNNEL_TOKEN", args.token.strip())
        manager.token = args.token.strip()
        print("\n🎉 Token saved! Testing your permanent tunnel connection now...")
        test_quick_tunnel(manager)
        return

    if args.domain:
        dom = args.domain.strip()
        if not dom.startswith("http"):
            dom = f"https://{dom}"
        update_env_file("STATIC_DOMAIN", dom)
        manager.static_domain = dom
        print(f"\n🎉 Static domain saved as: {dom}")
        return

    if args.duckdns:
        dom = args.duckdns.strip()
        if not dom.startswith("http"):
            dom = f"http://{dom}:8000" if ":" not in dom else f"http://{dom}"
        update_env_file("STATIC_DOMAIN", dom)
        print(f"\n🎉 DuckDNS / DDNS static domain saved as: {dom}")
        return

    if args.wizard:
        verify_binary(manager)
        run_interactive_wizard(manager)
        return
    
    if args.check or args.download:
        verify_binary(manager, force_download=args.download)

    if args.test_quick:
        verify_binary(manager)
        test_quick_tunnel(manager)


if __name__ == "__main__":
    main()
