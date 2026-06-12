"""Checkpoint helpers: survive Colab disconnects by saving to Drive (Phase 1).

Uses Flax's msgpack serialization for params/optimizer state. Point ``base_dir``
at a mounted Drive folder (e.g. ``/content/drive/MyDrive/craftax-latent-action``)
so checkpoints persist across runtime restarts.
"""

from __future__ import annotations

import os
import pickle

from flax import serialization


def save_state(state, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(state))
    print(f"[ckpt] saved -> {path}")


def load_state(state, path: str):
    """Load bytes into a matching ``state`` template (same structure)."""
    with open(path, "rb") as f:
        return serialization.from_bytes(state, f.read())


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[ckpt] saved -> {path}")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def mount_drive(base_subdir: str = "craftax-latent-action") -> str:
    """Mount Google Drive (Colab only) and return a project base directory.

    Falls back to a local directory off-Colab so the same code runs anywhere.
    """
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        base = f"/content/drive/MyDrive/{base_subdir}"
    except Exception:
        base = os.path.abspath(base_subdir)
    os.makedirs(base, exist_ok=True)
    for sub in ("checkpoints", "data/trajectories", "results"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    print(f"[ckpt] project base = {base}")
    return base
