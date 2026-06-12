"""Trajectory dataset collection (Phase 1).

Roll a policy (random and/or a trained PPO actor) through the *pixel* Craftax
env and save ``(frames, actions, rewards, dones)`` to disk. The latent-action
model (Phase 2) consumes the frames; the action labels are held out and only
used for validation and the small labelled-decoder budgets (Phase 4).

Stored as a compressed ``.npz`` with uint8 frames to keep size manageable
(63x63x3 uint8 ~= 12KB/frame; 1M frames ~= 12GB, so default targets are
smaller -- bump ``total_frames`` if you have the Drive space).
"""

from __future__ import annotations

import os
import functools
from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .env_utils import make_craftax, get_dims, reset_vec, step_vec


def _step_once(state, policy, env, env_params, is_pixels):
    """One env step producing a host-friendly transition dict (used by the
    chunked collector's scan body). Frames are quantized to uint8 on-device for
    pixel mode so the host copy is 4x smaller."""
    p_rng = jax.random.fold_in(state.rng, 1)
    actions = policy(state.obs, p_rng)
    new_state, tr = step_vec(env, env_params, state, actions)
    frame = tr["obs"]
    if is_pixels:
        frame = jnp.clip(jnp.round(frame * 255.0), 0, 255).astype(jnp.uint8)
    out = {
        "frame": frame,
        "action": tr["action"].astype(jnp.int16),
        "reward": tr["reward"].astype(jnp.float32),
        "done": tr["done"],
    }
    return new_state, out


def _random_policy(num_actions):
    def policy(obs, rng):
        return jax.random.randint(rng, (obs.shape[0],), 0, num_actions)
    return policy


def _ppo_policy(train_state, net_apply, epsilon: float = 0.0):
    """Greedy-ish PPO policy. ``epsilon`` mixes in random actions to widen
    state coverage in the collected dataset."""
    def policy(obs, rng):
        pi, _ = net_apply(train_state.params, obs)
        a_rng, e_rng, u_rng = jax.random.split(rng, 3)
        action = pi.sample(seed=a_rng)
        rand = jax.random.randint(u_rng, action.shape, 0, pi.logits.shape[-1])
        mix = jax.random.uniform(e_rng, action.shape) < epsilon
        return jnp.where(mix, rand, action)
    return policy


def collect_dataset(
    policy: Callable,
    total_frames: int = 500_000,
    num_envs: int = 256,
    obs_mode: str = "pixels",
    seed: int = 0,
    save_path: Optional[str] = None,
    chunk_steps: int = 64,
):
    """Collect ``total_frames`` transitions under ``policy``.

    ``policy(obs, rng) -> actions`` maps a batch of observations to a batch of
    discrete actions. Returns a dict of numpy arrays and optionally saves it.

    The rollout is done in chunks of ``chunk_steps`` env-steps: each chunk is
    converted to uint8 and copied to host RAM before the next chunk runs, so GPU
    memory only ever holds one chunk's worth of frames (not the whole dataset).
    This is what keeps pixel collection from OOM-ing on a single GPU -- a full
    300k-frame float32 buffer is ~13 GB, but one chunk is a few hundred MB.
    """
    env, env_params = make_craftax(obs_mode=obs_mode, auto_reset=True)
    _, num_actions = get_dims(env, env_params)

    total_steps = total_frames // num_envs
    chunk_steps = min(chunk_steps, total_steps)
    num_chunks = (total_steps + chunk_steps - 1) // chunk_steps
    rng = jax.random.PRNGKey(seed)
    state = reset_vec(env, env_params, rng, num_envs)

    is_pixels = obs_mode == "pixels"

    @functools.partial(jax.jit, static_argnums=(1,))
    def rollout_chunk(state, length):
        return jax.lax.scan(
            lambda s, _: _step_once(s, policy, env, env_params, is_pixels),
            state, None, length=length)

    frames_chunks, action_chunks, reward_chunks, done_chunks = [], [], [], []
    remaining = total_steps
    for c in range(num_chunks):
        length = min(chunk_steps, remaining)
        state, traj = rollout_chunk(state, length)
        # pull this chunk to host RAM, freeing GPU memory before the next chunk
        frames_chunks.append(np.asarray(traj["frame"]))
        action_chunks.append(np.asarray(traj["action"]))
        reward_chunks.append(np.asarray(traj["reward"]))
        done_chunks.append(np.asarray(traj["done"]))
        remaining -= length
        print(f"[collect] chunk {c + 1}/{num_chunks}  "
              f"({(total_steps - remaining) * num_envs:,}/{total_frames:,} frames)")

    def stack_flat(chunks):
        x = np.concatenate(chunks, axis=0)          # (steps, N, ...)
        return x.reshape((-1,) + x.shape[2:])       # (steps*N, ...)

    frames = stack_flat(frames_chunks)
    if not is_pixels:  # symbolic obs stay float; leave as-is
        frames = frames.astype(np.float32)

    data = {
        "frames": frames,
        "actions": stack_flat(action_chunks).astype(np.int16),
        "rewards": stack_flat(reward_chunks).astype(np.float32),
        "dones": stack_flat(done_chunks).astype(bool),
        "num_actions": np.int64(num_actions),
    }

    print(f"[collect] frames={data['frames'].shape} dtype={data['frames'].dtype} "
          f"actions in [{data['actions'].min()}, {data['actions'].max()}] "
          f"size~={data['frames'].nbytes / 1e9:.2f}GB")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.savez_compressed(save_path, **data)
        print(f"[collect] saved -> {save_path}")
    return data


def make_pair_dataset(data: dict, drop_episode_boundaries: bool = True):
    """Turn a flat transition dump into (o_t, o_{t+1}, a_t) pairs for the
    latent-action model. Optionally drops pairs that straddle an episode reset
    (where ``done[t]`` is True), since (o_t, o_{t+1}) is meaningless there."""
    frames = data["frames"]
    actions = data["actions"]
    dones = data["dones"]

    o_t = frames[:-1]
    o_tp1 = frames[1:]
    a_t = actions[:-1]
    valid = ~dones[:-1] if drop_episode_boundaries else np.ones(len(a_t), bool)

    o_t, o_tp1, a_t = o_t[valid], o_tp1[valid], a_t[valid]
    print(f"[pairs] {len(a_t):,} valid (o_t, o_t+1, a_t) pairs")
    return {"o_t": o_t, "o_tp1": o_tp1, "a_t": a_t}


def load_dataset(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}
