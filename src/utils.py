"""
utils.py
--------
Project-wide utility helpers.

Blueprint §23 (src/utils.py):
  Config loader, seed, logger, file I/O helpers.

Exports:
    load_config           — merge base.yaml + stage-specific YAML
    set_seed              — reproducible seed for torch / numpy / random
    get_logger            — named logger with standard formatter
    ensure_dir            — mkdir -p helper
    save_json / load_json — lightweight JSON I/O
    torch_device          — resolve "auto" → cuda/cpu torch.device
    count_parameters      — trainable param count for a nn.Module
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a named logger configured with the project-standard format.

    Parameters
    ----------
    name  : logger name (usually __name__ of the calling module)
    level : one of DEBUG | INFO | WARNING | ERROR

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(
    stage_config_path: str | Path | None = None,
    base_config_path: str | Path | None = None,
) -> dict:
    """Load and merge project configuration.

    Merges base.yaml with a stage-specific YAML.  Stage values override base
    values at the top-level key granularity (dict.update per section).

    Parameters
    ----------
    stage_config_path : path to stage-specific YAML (e.g. configs/cnn.yaml)
    base_config_path  : path to base YAML; defaults to configs/base.yaml
                        relative to this file's project root.

    Returns
    -------
    dict — merged configuration
    """
    # Resolve project root (pa-gnn/)
    project_root = Path(__file__).resolve().parent.parent

    if base_config_path is None:
        base_config_path = project_root / "configs" / "base.yaml"

    cfg: dict = {}

    # Load base
    base_path = Path(base_config_path)
    if base_path.exists():
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        cfg.update(base_cfg)

    # Load and merge stage-specific
    if stage_config_path is not None:
        stage_path = Path(stage_config_path)
        if not stage_path.is_absolute():
            stage_path = project_root / stage_path
        if stage_path.exists():
            with open(stage_path) as f:
                stage_cfg = yaml.safe_load(f) or {}
            # Deep-merge at top-level section level
            for key, val in stage_cfg.items():
                if key in cfg and isinstance(cfg[key], dict) and isinstance(val, dict):
                    cfg[key] = {**cfg[key], **val}
                else:
                    cfg[key] = val
        else:
            logging.warning("Stage config not found: %s", stage_path)

    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed : integer seed value (blueprint default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops — slightly slower but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def torch_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to a torch.device.

    Parameters
    ----------
    device_str : "auto" | "cuda" | "cpu" | "cuda:N"

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Parameters
    ----------
    path : directory path

    Returns
    -------
    Path object of the created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise data to JSON file.

    Parameters
    ----------
    data  : JSON-serialisable object
    path  : output file path (parent dirs created automatically)
    indent: JSON pretty-print indent
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load JSON file.

    Parameters
    ----------
    path : file path

    Returns
    -------
    Parsed JSON object.

    Raises
    ------
    FileNotFoundError if path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a nn.Module.

    Parameters
    ----------
    model          : PyTorch module
    trainable_only : if True, count only requires_grad parameters

    Returns
    -------
    int — total parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_param_count(model: torch.nn.Module) -> str:
    """Return human-readable parameter count string, e.g. '11.7 M'."""
    n = count_parameters(model)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


# ---------------------------------------------------------------------------
# Split file helpers
# ---------------------------------------------------------------------------

def read_split_file(path: str | Path) -> list[str]:
    """Read a split text file (one tile stem per line) into a list.

    Parameters
    ----------
    path : e.g. data/splits/train.txt

    Returns
    -------
    list of stripped non-empty lines
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    lines = p.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def write_split_file(stems: list[str], path: str | Path) -> None:
    """Write a list of tile stems to a split text file.

    Parameters
    ----------
    stems : list of tile stem strings
    path  : output file path (parent dirs created automatically)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(stems) + "\n")
