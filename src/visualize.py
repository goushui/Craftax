"""Visualization helpers: watch Craftax gameplay play out (any phase).

Three things you can render:
  1. A rollout under any policy -> frame grid, animated GIF, or inline video.
  2. The latent-action model's forward prediction next to ground truth (Phase 2).
  3. A world-model imagination rollout next to the real env (Phase 3/4).

Craftax exposes two image sources:
  * the *pixel observation* (63x63, what the agent sees) -- always available;
  * the *high-res renderer* ``render_craftax_pixels`` (crisper, larger tiles) --
    nicer for GIFs/figures. We try the renderer and fall back to the obs.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .env_utils import make_craftax, get_dims, reset_vec, step_vec


# ---------------------------------------------------------------------------
# Frame normalization
# ---------------------------------------------------------------------------

def _to_uint8(frame) -> np.ndarray:
    f = np.asarray(frame)
    if f.dtype != np.uint8:
        if f.max() <= 1.0 + 1e-3:
            f = f * 255.0
        f = f.clip(0, 255).astype(np.uint8)
    return f


def _hi_res_renderer(block_pixel_size: int = 16):
    """Return a ``render(env_state) -> HxWx3`` fn using Craftax's renderer, or
    None if unavailable in this Craftax version."""
    try:
        from craftax.craftax_classic.renderer import render_craftax_pixels
        def render(env_state):
            img = render_craftax_pixels(env_state, block_pixel_size=block_pixel_size)
            return _to_uint8(img)
        return render
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rollout -> frames
# ---------------------------------------------------------------------------

def rollout_frames(policy: Optional[Callable] = None, num_steps: int = 256,
                   seed: int = 0, hi_res: bool = True, block_pixel_size: int = 16):
    """Roll a *single* env and return ``(frames, actions, rewards)``.

    ``policy(obs, rng) -> action`` for a batch of 1; if None, acts randomly.
    Uses the high-res renderer when available, else the pixel observation.
    """
    env, env_params = make_craftax("pixels", auto_reset=True)
    _, num_actions = get_dims(env, env_params)
    if policy is None:
        def policy(obs, rng):
            return jax.random.randint(rng, (obs.shape[0],), 0, num_actions)

    render = _hi_res_renderer(block_pixel_size) if hi_res else None

    rng = jax.random.PRNGKey(seed)
    state = reset_vec(env, env_params, rng, num_envs=1)

    frames, actions, rewards = [], [], []
    for _ in range(num_steps):
        if render is not None:
            # env_state is a batched pytree (N=1); index 0 for the renderer
            single = jax.tree_util.tree_map(lambda x: x[0], state.env_state)
            frames.append(render(single))
        else:
            frames.append(_to_uint8(state.obs[0]))

        p_rng = jax.random.fold_in(state.rng, 1)
        action = policy(state.obs, p_rng)
        state, tr = step_vec(env, env_params, state, action)
        actions.append(int(np.asarray(action)[0]))
        rewards.append(float(np.asarray(tr["reward"])[0]))

    return np.stack(frames), np.array(actions), np.array(rewards)


def ppo_policy_single(train_state, net_apply, greedy: bool = False):
    """Wrap a trained PPO state into a single-env policy for ``rollout_frames``.
    Note: PPO was trained on *symbolic* obs, so this only works if you roll the
    symbolic env. For pixel rollouts, pass ``policy=None`` (random) to just watch
    the world, or train a pixel PPO."""
    def policy(obs, rng):
        pi, _ = net_apply(train_state.params, obs)
        return jnp.argmax(pi.logits, -1) if greedy else pi.sample(seed=rng)
    return policy


# ---------------------------------------------------------------------------
# Display: GIF, video, grid
# ---------------------------------------------------------------------------

def save_gif(frames, path: str, fps: int = 8, scale: int = 4):
    """Write frames to an animated GIF (nearest-neighbour upscaled by ``scale``)."""
    import imageio
    big = [np.kron(f, np.ones((scale, scale, 1), dtype=f.dtype)) for f in frames]
    imageio.mimsave(path, big, fps=fps)
    print(f"[viz] saved GIF -> {path} ({len(frames)} frames)")
    return path


def show_video(frames, fps: int = 8, scale: int = 4):
    """Inline HTML5 video in a notebook (smoother than a GIF for long rollouts)."""
    import imageio
    from IPython.display import Video, display
    import tempfile, os
    big = [np.kron(f, np.ones((scale, scale, 1), dtype=f.dtype)) for f in frames]
    tmp = os.path.join(tempfile.gettempdir(), "craftax_rollout.mp4")
    imageio.mimsave(tmp, big, fps=fps, codec="libx264")
    display(Video(tmp, embed=True, width=scale * frames.shape[2]))
    return tmp


def show_grid(frames, cols: int = 8, max_frames: int = 32, titles=None):
    """Plot a grid of frames (quick static look at a rollout)."""
    import matplotlib.pyplot as plt
    n = min(len(frames), max_frames)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(frames[i])
            if titles is not None:
                axes[i].set_title(str(titles[i]), fontsize=7)
    plt.tight_layout()
    return fig


def quick_watch(num_steps: int = 200, fps: int = 10, seed: int = 0,
                as_video: bool = True, save_path: Optional[str] = None):
    """One-liner: roll a random policy and show it inline. Great first sanity check."""
    frames, actions, rewards = rollout_frames(num_steps=num_steps, seed=seed)
    print(f"[viz] {len(frames)} frames, total reward = {rewards.sum():.2f}")
    if save_path:
        save_gif(frames, save_path, fps=fps)
    if as_video:
        try:
            return show_video(frames, fps=fps)
        except Exception as e:
            print(f"[viz] video failed ({e}); falling back to grid")
    return show_grid(frames)


# ---------------------------------------------------------------------------
# Phase 2: latent-action reconstruction (predicted o_{t+1} vs. real)
# ---------------------------------------------------------------------------

def show_lam_predictions(lam_model, lam_state, pairs, n: int = 6, seed: int = 0):
    """For ``n`` random pairs show o_t, true o_{t+1}, and the LAM's predicted
    o_{t+1} -- a visual read on whether the latent action captured the change."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs["o_t"]), size=n)
    o_t = pairs["o_t"][idx].astype(np.float32) / 255.0
    o_tp1 = pairs["o_tp1"][idx].astype(np.float32) / 255.0
    out = lam_model.apply(lam_state.params, jnp.asarray(o_t), jnp.asarray(o_tp1))
    pred = np.asarray(out["recon"])
    codes = np.asarray(out["codes"])

    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 5.4))
    for j in range(n):
        axes[0, j].imshow(o_t[j]); axes[0, j].set_title(f"o_t", fontsize=8)
        axes[1, j].imshow(o_tp1[j]); axes[1, j].set_title("o_t+1 (true)", fontsize=8)
        axes[2, j].imshow(pred[j].clip(0, 1))
        axes[2, j].set_title(f"pred (z={codes[j]})", fontsize=8)
        for r in range(3):
            axes[r, j].axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Phase 3/4: world-model imagination vs. reality (symbolic obs -> not directly
# an image; this renders the *real* env frames the actions produce, with the
# RSSM's predicted reward/continue overlaid as a sanity check).
# ---------------------------------------------------------------------------

def imagination_vs_real_rewards(rssm_model, rssm_state, rssm_cfg,
                                obs, actions, rewards, dones, horizon=20, seed=0):
    """Plot RSSM-predicted reward vs. true reward over an open-loop rollout.
    A dynamics analogue of 'watching it play' for the symbolic world model."""
    import matplotlib.pyplot as plt
    from .rssm import make_sequence_sampler
    sampler = make_sequence_sampler(obs, actions, rewards, dones,
                                    rssm_cfg.num_inputs, horizon + 8, seed)
    batch = sampler(64)
    context = 8

    def run(params, batch, rng):
        _, _, _, (h, stoch) = rssm_model.apply(
            params, batch["obs"][:context], batch["action"][:context], rng,
            method=rssm_model.observe)
        def step(carry, act_t):
            h, stoch, rng = carry
            rng, r = jax.random.split(rng)
            h, stoch, feat = rssm_model.apply(params, h, stoch, act_t, r,
                                              method=rssm_model.imagine_step)
            return (h, stoch, rng), rssm_model.apply(params, feat,
                                                     method=rssm_model.decode_reward)
        fut = batch["action"][context:context + horizon]
        _, preds = jax.lax.scan(step, (h, stoch, rng), fut)
        return preds

    pred = np.asarray(jax.jit(run)(rssm_state.params, batch, jax.random.PRNGKey(seed)))
    true = np.asarray(batch["reward"][context:context + horizon])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pred.mean(1), "-o", label="RSSM predicted reward")
    ax.plot(true.mean(1), "-s", label="true reward")
    ax.set_xlabel("imagination step"); ax.set_ylabel("mean reward")
    ax.set_title("World-model imagination vs. reality"); ax.legend(); ax.grid(alpha=0.3)
    return fig
