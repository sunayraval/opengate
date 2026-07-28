# 🌐 Cloudflare Tunnel Setup & 24/7 Permanent URL Configuration

While the standard `README.md` explains how to run this server locally on your home network, you may want your robot to operate entirely autonomously over cellular data (4G/5G) or outside of your home Wi-Fi network. 

To achieve this without configuring router port-forwarding or setting up VPNs, we've integrated **Cloudflare Zero-Trust Tunnels (`cloudflared`)**. This automatically creates an encrypted outbound HTTPS/WSS tunnel from your Desktop to the internet.

---

## 🔒 Enabling Cloudflare in the Backend

Before setting up the domain, you need to tell the server to enable Cloudflare routing.

1. Open your `.env` file (or `app/config.py`).
2. Set the `ENABLE_CLOUDFLARE` setting to `True`:
   ```env
   ENABLE_CLOUDFLARE=True
   ```
3. When you start your server with `uv run runner.py`, it will now spawn a Cloudflare tunnel.

If you don't configure a token (explained below), the system creates a **Quick Tunnel (Dev Mode)** with a randomized URL that changes every time your PC server restarts. 

For true **24/7 Robot Autonomy**, you want a **Permanent Static Domain** (e.g., `https://api.myrobot.com`) so you can hardcode it into your Raspberry Pi once and never worry about server reboots!

---

## 🧙 Option 1: Automated Setup Wizard (Recommended & 100% Free)

We built an interactive wizard that configures your permanent static domain in seconds:

```powershell
python scripts/setup_cloudflare.py --wizard
```

**What the wizard does:**
1. Guides you on where to copy your Free Tunnel Token from the [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com) (*Networks -> Tunnels -> Create Tunnel*).
2. Prompts you to paste your Token and optional Custom Domain.
3. Automatically writes your credentials into a `.env` file in the project root.
4. Launches a live test connection to verify that your permanent domain is active!

---

## 🛠️ Option 2: One-Line CLI Command 

If you already have your Cloudflare Tunnel Token, you can configure it instantly via the command line:

```powershell
# Save Cloudflare Tunnel Token & test connection automatically:
python scripts/setup_cloudflare.py --token "eyJhIjoi..." --domain "https://api.myrobot.com"
```

Once configured, your permanent URL is loaded automatically every time you run the server. **You can now safely hardcode this URL into your Raspberry Pi 5!**

---

## 🧪 Troubleshooting & Diagnostics

If you ever experience connection issues or want to check your tunnel binary status, run our diagnostic tool:

```powershell
# Check cloudflared binary version and status:
python scripts/setup_cloudflare.py --check

# Test launching a quick ephemeral tunnel:
python scripts/setup_cloudflare.py --test-quick

# Display full networking guide:
python scripts/setup_cloudflare.py --info
```
