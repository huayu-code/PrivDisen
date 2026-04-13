"""
Configuration management using YAML + command-line overrides.
"""

import argparse
import os
from typing import Any, Dict, Optional

import yaml


class Config:
    """Simple hierarchical config backed by a dict."""

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None):
        self._cfg = cfg_dict or {}

    # ------------------------------------------------------------------
    # dict-like access
    # ------------------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        val = self._cfg.get(key)
        if isinstance(val, dict):
            return Config(val)
        if val is None:
            raise AttributeError(f"Config has no attribute '{key}'")
        return val

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: str) -> bool:
        return key in self._cfg

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return self._cfg

    def __repr__(self) -> str:
        return f"Config({self._cfg})"

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        return cls(cfg_dict or {})

    def merge(self, overrides: Dict[str, Any]) -> "Config":
        """Return a new Config with *overrides* applied on top."""
        merged = {**self._cfg}
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v
        return Config(merged)


def parse_args() -> argparse.Namespace:
    """Shared CLI argument parser."""
    p = argparse.ArgumentParser(description="PrivDisen Experiments")

    # --- general ---
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--method", type=str, default="privdisen",
                   choices=["vanilla", "dp_vfl", "svfl", "labobf", "kdk",
                            "ladsg", "mid", "privdisen"],
                   help="Defense method to use")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    # --- data ---
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--num_parties", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)

    # --- training ---
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    # --- PrivDisen specific ---
    p.add_argument("--task_dim", type=int, default=None)
    p.add_argument("--private_dim", type=int, default=None)
    p.add_argument("--beta", type=float, default=None,
                   help="MI constraint strength (privacy knob)")
    p.add_argument("--gamma", type=float, default=None,
                   help="Reconstruction loss weight")
    p.add_argument("--delta", type=float, default=None,
                   help="HSIC independence weight")
    p.add_argument("--alpha_schedule", type=str, default=None,
                   choices=["dann", "linear", "constant"])

    # --- evaluation ---
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--attacks", nargs="+", default=None,
                   choices=["norm", "direction", "model_completion",
                            "embedding_extension"])

    return p.parse_args()


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Load YAML config and merge CLI overrides."""
    if args is None:
        args = parse_args()

    config_path = args.config
    if os.path.exists(config_path):
        cfg = Config.from_yaml(config_path)
    else:
        cfg = Config()

    # Merge CLI overrides (skip None values)
    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config" and v is not None}
    cfg = cfg.merge(cli_overrides)

    # 自动检测设备：device="auto" 时自动选择 cuda 或 cpu
    device = cfg.get("device", "auto")
    if device == "auto":
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        cfg = cfg.merge({"device": device})

    return cfg
