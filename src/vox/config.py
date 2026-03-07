"""Persistent configuration for vox."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "vox"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_MODEL_PATH = str(Path.home() / "models" / "Voxtral-Mini-4B-Realtime-6bit")


VALID_MODES = ("transcript", "realtime")
VALID_BACKENDS = ("voxtral", "parakeet", "moonshine")

DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_MOONSHINE_ARCH = "medium-streaming"


@dataclass
class Config:
    model_path: str = DEFAULT_MODEL_PATH
    post_processing: bool = True
    type_at_cursor: bool = False
    sample_rate: int = 16_000
    mode: str = "transcript"  # "transcript" | "realtime"
    backend: str = "voxtral"  # "voxtral" | "parakeet" | "moonshine"
    parakeet_model: str = DEFAULT_PARAKEET_MODEL
    moonshine_arch: str = DEFAULT_MOONSHINE_ARCH
    dev_mode: bool = False  # show all backends in menu; env VOX_DEV=1 overrides

    @property
    def is_dev(self) -> bool:
        """True if dev mode is enabled (config or VOX_DEV=1 env var)."""
        return self.dev_mode or os.environ.get("VOX_DEV") == "1"

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode {self.mode!r}, must be one of {VALID_MODES}"
            )
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"Invalid backend {self.backend!r}, must be one of {VALID_BACKENDS}"
            )
        # Non-dev mode: force voxtral backend.
        if not self.is_dev and self.backend != "voxtral":
            self.backend = "voxtral"

    @classmethod
    def load(cls) -> Config:
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
                return cls(**known)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2) + "\n")
