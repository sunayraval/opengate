import logging
import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback if pydantic_settings is not installed independently from pydantic v1/v2
    from pydantic import BaseSettings

import torch

logger = logging.getLogger("app.config")


class Settings(BaseSettings):
    """
    Application configuration settings for the PC-side Real-Time AI CV Robot Framework.
    
    Settings can be configured via environment variables or a .env file located in the
    workspace root directory.
    """
    # Network & Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Cloudflare & DNS Networking Settings
    ENABLE_CLOUDFLARE: bool = False
    CLOUDFLARE_TUNNEL_TOKEN: Optional[str] = None
    STATIC_DOMAIN: Optional[str] = None

    # Hardware & Precision Settings
    USE_FP16: bool = True

    # AI Model Defaults
    DEFAULT_COMPLETION_MODEL: str = "openbmb/MiniCPM-V"
    WEIGHTS_DIR: str = "./weights"

    # Pydantic Settings configuration
    if hasattr(BaseSettings, "model_config"):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=True,
            extra="ignore",
        )
    else:
        # Pydantic v1 fallback
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True

    def __init__(self, **values):
        super().__init__(**values)
        self._validate_and_setup()

    def _validate_and_setup(self) -> None:
        """
        Validates hardware availability and creates necessary local directories.
        Automatically falls back USE_FP16 to False if CUDA is not available.
        """
        # Ensure weights directory exists
        weights_path = Path(self.WEIGHTS_DIR)
        try:
            weights_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create WEIGHTS_DIR at {weights_path}: {e}")

        # Auto-fallback USE_FP16 if CUDA is unavailable
        if self.USE_FP16 and not torch.cuda.is_available():
            logger.warning(
                "USE_FP16 is set to True, but CUDA (NVIDIA GPU) is not available on this system. "
                "Automatically falling back USE_FP16 to False (using FP32 CPU inference)."
            )
            self.USE_FP16 = False
        elif self.USE_FP16 and torch.cuda.is_available():
            logger.info(f"CUDA available ({torch.cuda.get_device_name(0)}). FP16 half-precision inference enabled.")
        else:
            logger.info("FP16 disabled by configuration or CPU mode active.")


# Global singleton configuration instance
config = Settings()
