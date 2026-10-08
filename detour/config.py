"""Load a run config. A config may name a `base:` config; its own keys then override the base's keys.

Side checks use this: configs/s0_side_placement.yaml is the main config with one option changed.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base_path = (path.parent / cfg.pop("base")).resolve()
        cfg = _merge(load_config(base_path), cfg)
    return cfg


def config_files(path):
    """The config file and every base it extends (for copying into the run folder)."""
    path = Path(path).resolve()
    files = [path]
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        files += config_files(path.parent / cfg["base"])
    return files
