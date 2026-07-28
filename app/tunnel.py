import atexit
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("app.tunnel")

# Official Windows x86_64 release URL for cloudflared
CLOUDFLARED_WIN_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
CLOUDFLARED_LINUX_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
CLOUDFLARED_MAC_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"


class CloudflareTunnelManager:
    """
    Robust manager for Cloudflare Tunnels (cloudflared).
    
    Supports two operating modes:
    1. Quick Tunnel (Dev/Zero Account Mode): Launches an anonymous ephemeral tunnel and automatically
       parses the public https://xxxx.trycloudflare.com URL from process logs.
    2. Zero Trust Static Tunnel (Prod Mode): Uses a provided CLOUDFLARE_TUNNEL_TOKEN to connect
       to a pre-configured DNS hostname in Cloudflare Zero Trust dashboard.
       
    Automatically detects if cloudflared binary is installed in PATH or local ./bin directory.
    If missing, downloads the official binary from GitHub automatically.
    """

    def __init__(
        self,
        port: int = 8000,
        token: Optional[str] = None,
        static_domain: Optional[str] = None,
        bin_dir: str = "./bin",
    ):
        self.port = port
        self.token = token
        self.static_domain = static_domain
        self.bin_dir = Path(bin_dir)
        self.process: Optional[subprocess.Popen] = None
        self.tunnel_url: Optional[str] = None
        self._url_found_event = threading.Event()
        self._stop_event = threading.Event()
        self.executable_path: Optional[Path] = None

        # Register cleanup on exit
        atexit.register(self.stop)

    def _get_expected_binary_name(self) -> str:
        """Returns the OS-specific binary filename."""
        return "cloudflared.exe" if platform.system().lower() == "windows" else "cloudflared"

    def _get_download_url(self) -> str:
        """Returns the official GitHub download URL for the current operating system."""
        sys_name = platform.system().lower()
        if sys_name == "windows":
            return CLOUDFLARED_WIN_URL
        elif sys_name == "darwin":
            return CLOUDFLARED_MAC_URL
        else:
            return CLOUDFLARED_LINUX_URL

    def download_cloudflared(self) -> Path:
        """
        Downloads the official cloudflared executable from GitHub into the local bin directory.
        """
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        bin_name = self._get_expected_binary_name()
        target_path = self.bin_dir / bin_name
        download_url = self._get_download_url()

        logger.info(f"cloudflared not found in PATH. Downloading from {download_url} to {target_path}...")
        print(f"\n[CloudflareTunnelManager] Downloading official cloudflared binary to {target_path}...")
        print("This only happens once during initial setup.\n")

        try:
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Ensure executable permissions on Unix systems
            if platform.system().lower() != "windows":
                os.chmod(target_path, 0o755)

            logger.info(f"Successfully downloaded cloudflared to {target_path}")
            print(f"[CloudflareTunnelManager] Download complete! Binary ready at: {target_path}\n")
            return target_path

        except Exception as e:
            logger.error(f"Failed to download cloudflared: {e}")
            print(f"[ERROR] Failed to download cloudflared binary: {e}", file=sys.stderr)
            raise RuntimeError(f"Could not download cloudflared: {e}") from e

    def find_or_download_cloudflared(self) -> Path:
        """
        Locates cloudflared in PATH or local directory. If missing, downloads it automatically.
        """
        bin_name = self._get_expected_binary_name()

        # 1. Check system PATH
        system_bin = shutil.which(bin_name)
        if system_bin:
            logger.info(f"Found system cloudflared binary at: {system_bin}")
            self.executable_path = Path(system_bin)
            return self.executable_path

        # 2. Check local ./bin directory
        local_bin = self.bin_dir / bin_name
        if local_bin.exists() and local_bin.is_file():
            logger.info(f"Found local cloudflared binary at: {local_bin}")
            self.executable_path = local_bin
            return self.executable_path

        # 3. Check project root directory
        root_bin = Path(".") / bin_name
        if root_bin.exists() and root_bin.is_file():
            logger.info(f"Found cloudflared in project root at: {root_bin}")
            self.executable_path = root_bin
            return self.executable_path

        # 4. If not found anywhere, download automatically
        self.executable_path = self.download_cloudflared()
        return self.executable_path

    def _monitor_stream(self, stream, stream_name: str) -> None:
        """
        Background thread worker to read subprocess stdout/stderr line by line.
        Parses Quick Tunnel URL and logs tunnel activity.
        """
        url_regex = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
        
        try:
            for line in iter(stream.readline, ""):
                if self._stop_event.is_set():
                    break
                if not line:
                    continue

                line_str = line.strip()
                logger.debug(f"[cloudflared {stream_name}] {line_str}")

                # Check for trycloudflare.com URL in Quick Tunnel mode
                if not self.tunnel_url:
                    match = url_regex.search(line_str)
                    if match:
                        self.tunnel_url = match.group(0)
                        self._url_found_event.set()
                        self._print_tunnel_banner(self.tunnel_url)
        except ValueError:
            # Stream closed
            pass
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Error reading cloudflared {stream_name}: {e}")

    def _print_tunnel_banner(self, url: str) -> None:
        """Prints a prominent ASCII banner displaying the public tunnel URL."""
        banner = (
            "\n" + "=" * 76 + "\n"
            "   CLOUDFLARE TUNNEL ESTABLISHED SUCCESSFULLY!\n"
            "   \n"
            f"   PUBLIC AI API URL : {url}\n"
            f"   WS STREAMING URL  : {url.replace('https://', 'wss://')}/api/v1/stream\n"
            "   \n"
            "   Configure your Raspberry Pi / Robot client to connect to the above URLs.\n"
            + "=" * 76 + "\n"
        )
        print(banner)
        logger.info(f"Cloudflare Tunnel active at: {url}")

    def start(self, timeout_sec: int = 20) -> Optional[str]:
        """
        Starts the Cloudflare tunnel in the background.
        
        Returns:
            The public tunnel URL (for Quick Tunnels) or static domain (for token mode).
        """
        if self.process and self.process.poll() is None:
            logger.warning("Cloudflare tunnel is already running.")
            return self.tunnel_url

        exe_path = self.find_or_download_cloudflared()
        self._stop_event.clear()
        self._url_found_event.clear()

        # Build command line arguments
        if self.token and self.token.strip():
            logger.info("Starting Cloudflare Tunnel in Static Token Mode...")
            cmd = [str(exe_path), "tunnel", "run", "--token", self.token.strip()]
            self.tunnel_url = self.static_domain or "Static Zero-Trust Tunnel (Check Cloudflare Dashboard)"
            self._url_found_event.set()
        else:
            logger.info(f"Starting Cloudflare Quick Tunnel forwarding to http://localhost:{self.port}...")
            cmd = [
                str(exe_path),
                "tunnel",
                "--url",
                f"http://localhost:{self.port}",
                "--no-autoupdate",
            ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system().lower() == "windows" else 0,
            )

            # Start background monitoring threads for stdout and stderr
            t_stdout = threading.Thread(
                target=self._monitor_stream,
                args=(self.process.stdout, "stdout"),
                daemon=True,
                name="cloudflared-stdout"
            )
            t_stderr = threading.Thread(
                target=self._monitor_stream,
                args=(self.process.stderr, "stderr"),
                daemon=True,
                name="cloudflared-stderr"
            )
            t_stdout.start()
            t_stderr.start()

            # In Quick Tunnel mode, wait until URL is parsed from logs
            if not (self.token and self.token.strip()):
                logger.info(f"Waiting up to {timeout_sec}s for Cloudflare Quick Tunnel URL assignment...")
                url_found = self._url_found_event.wait(timeout=timeout_sec)
                if not url_found:
                    logger.warning("Timeout waiting for trycloudflare.com URL. Check logs or internet connectivity.")
                    print("[WARNING] Cloudflare tunnel started, but public URL was not detected within timeout.")
            else:
                self._print_tunnel_banner(self.tunnel_url)

            return self.tunnel_url

        except Exception as e:
            logger.error(f"Failed to launch cloudflared process: {e}")
            self.stop()
            raise RuntimeError(f"Could not start Cloudflare tunnel: {e}") from e

    def stop(self) -> None:
        """
        Gracefully terminates the cloudflared background process.
        """
        if not self.process:
            return

        self._stop_event.set()
        if self.process.poll() is None:
            logger.info("Stopping Cloudflare tunnel process...")
            try:
                self.process.terminate()
                # Wait up to 3 seconds for graceful shutdown
                for _ in range(30):
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.1)
                
                # Force kill if still running
                if self.process.poll() is None:
                    logger.warning("cloudflared did not terminate gracefully. Forcing kill...")
                    self.process.kill()
            except Exception as e:
                logger.error(f"Error while stopping cloudflared: {e}")
        
        self.process = None
        self.tunnel_url = None
        logger.info("Cloudflare tunnel stopped cleanly.")
