#!/usr/bin/env python3
"""
Automated Raspberry Pi Client Deployment Script.

This tool securely copies the minimal Raspberry Pi client codebase (`client_rpi/` directory)
from your Windows Desktop PC to your Raspberry Pi 5 over your local network using OpenSSH (SCP).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
client_dir = workspace_root / "client_rpi"


def print_banner():
    banner = """
================================================================================
             🚀 RASPBERRY PI 5 CLIENT DEPLOYMENT TOOL
================================================================================

This script will copy the lightweight robot client folder (`client_rpi/`) from your
Windows PC directly to your Raspberry Pi over your Wi-Fi or local Ethernet network!

Requirements:
  1. Your Raspberry Pi and Windows PC must be on the same local network (Wi-Fi/LAN).
  2. SSH must be enabled on your Raspberry Pi (run `sudo raspi-config` -> Interface Options -> SSH -> Yes).
================================================================================
"""
    print(banner)


def run_deployment(pi_host: str, pi_user: str, target_dir: str = "~/client_rpi"):
    if not client_dir.exists():
        print(f"[ERROR] Could not find client directory at: {client_dir}", file=sys.stderr)
        return False

    # Check if scp is available on Windows
    scp_bin = shutil.which("scp")
    if not scp_bin:
        print("[ERROR] 'scp' command not found in Windows PATH. Please ensure OpenSSH Client is installed in Windows Optional Features.", file=sys.stderr)
        print("\n💡 Alternative: You can manually drag-and-drop the 'client_rpi' folder to your Pi using WinSCP or FileZilla.")
        return False

    remote_dest = f"{pi_user}@{pi_host}:{target_dir}"
    print(f"\n📡 Initiating Secure Copy (SCP) to {remote_dest}...")
    print(f"Source Folder: {client_dir}\n")
    print("👉 Note: If prompted, enter your Raspberry Pi SSH password below:\n")

    cmd = [scp_bin, "-r", str(client_dir), remote_dest]

    try:
        res = subprocess.run(cmd, check=True)
        print("\n" + "="*76)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("="*76)
        print(f"\nThe 'client_rpi' folder is now installed on your Raspberry Pi at: {target_dir}")
        print("\n👉 NEXT STEPS ON YOUR RASPBERRY PI TERMINAL:")
        print("1. Open an SSH session or terminal on your Pi:")
        print(f"     ssh {pi_user}@{pi_host}")
        print("\n2. Install the lightweight Python dependencies (OpenCV + Requests):")
        print("     cd ~/client_rpi && pip install -r requirements_rpi.txt")
        print("\n3. Launch the AI Robot Client connecting to your PC's Cloudflare URL:")
        print("     python robot_client.py --url \"https://your-url.trycloudflare.com\" --mode action --actions \"move forward,turn left,stop\"")
        print("="*76 + "\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] SCP transfer failed with return code {e.returncode}.", file=sys.stderr)
        print("\n💡 TROUBLESHOOTING TIPS:")
        print("  - Check that the IP address or hostname is correct.")
        print("  - Verify SSH is enabled on the Pi (`sudo raspi-config` -> 3 Interface Options -> I2 SSH -> Yes).")
        print("  - Ensure both devices are connected to the same Wi-Fi network.")
        return False
    except KeyboardInterrupt:
        print("\n[ABORTED] Transfer cancelled by user.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy client_rpi directory to Raspberry Pi via SCP.")
    parser.add_argument("--host", "-ip", type=str, help="Raspberry Pi IP address or hostname (e.g. 192.168.1.50 or raspberrypi.local).")
    parser.add_argument("--user", "-u", type=str, default="pi", help="SSH username on the Raspberry Pi (default: pi or username you configured).")
    parser.add_argument("--dir", "-d", type=str, default="~/client_rpi", help="Destination path on the Raspberry Pi (default: ~/client_rpi).")
    
    args = parser.parse_args()

    print_banner()

    host = args.host
    user = args.user

    if not host:
        try:
            host_input = input("🌐 Enter Raspberry Pi IP or Hostname (default: raspberrypi.local): ").strip()
            host = host_input if host_input else "raspberrypi.local"
            
            user_input = input(f"👤 Enter Raspberry Pi SSH Username (default: {user}): ").strip()
            if user_input:
                user = user_input
        except KeyboardInterrupt:
            print("\n[ABORTED] Exiting.")
            return

    run_deployment(pi_host=host, pi_user=user, target_dir=args.dir)


if __name__ == "__main__":
    main()
