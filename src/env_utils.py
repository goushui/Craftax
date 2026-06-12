"""Craftax-classic environment wrappers and vectorization helpers (Phase 1).

Craftax-classic is a JAX-native re-implementation of Crafter. It exposes two
observation modes:

  * "symbolic"  -> a flat ~8268-dim vector (fast, used for the RL agent).
  * "pixels"    -> a 63x63x3 RGB image (used for the latent-action model, the
                   novel piece of this project).

Everything here is pure-JAX so it can be ``jax.jit`` / ``jax.vmap``-ed across
thousands of parallel envs, which is the entire point of using Craftax.

References
----------
Craftax: https://arxiv.org/pdf/2402.16801
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def make_craftax(obs_mode: str = "symbolic", auto_reset: bool = True):
    """Return ``(env, env_params)`` for Craftax-classic.

    Parameters
    ----------
    obs_mode:
        "symbolic" or "pixels". Selects which Craftax environment id to load.
    auto_reset:
        If True, use the auto-reset variant (env resets itself on ``done``),
        which is what you want for continuous PPO rollouts.
    """
    from craftax.craftax_env import make_craftax_env_from_name

    if obs_mode == "symbolic":
        name = "Craftax-Classic-Symbolic-AutoReset-v1" if auto_reset \
            else "Craftax-Classic-Symbolic-v1"
    elif obs_mode == "pixels":
        name = "Craftax-Classic-Pixels-AutoReset-v1" if auto_reset \
            else "Craftax-Classic-Pixels-v1"
    else:
        raise ValueError(f"Unknown obs_mode={obs_mode!r}")

    env = make_craftax_env_from_name(name, auto_reset=auto_reset)
    env_params = env.default_params
    return env, env_params


# ---------------------------------------------------------------------------
# Vectorized rollout state
# ---------------------------------------------------------------------------

class VecState(NamedTuple):
    """Carried state for a batch of parallel environments."""
    obs: jnp.ndarray          # (N, *obs_shape)
    env_state: Any            # pytree of per-env internal state
    rng: jax.Array


def reset_vec(env, env_params, rng: jax.Array, num_envs: int) -> VecState:
    """Reset ``num_envs`` environments in parallel."""
    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, num_envs)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    return VecState(obs=obs, env_state=env_state, rng=rng)


@functools.partial(jax.jit, static_argnums=(0,))
def step_vec(env, env_params, state: VecState, actions: jnp.ndarray):
    """Step a batch of environments.

    Returns ``(new_state, transition_dict)`` where the transition dict holds the
    pre-step obs, action, reward, done and next obs -- the fields you need for
    both PPO and trajectory dataset collection.
    """
    rng, step_rng = jax.random.split(state.rng)
    step_rngs = jax.random.split(step_rng, actions.shape[0])
    next_obs, next_env_state, reward, done, info = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(step_rngs, state.env_state, actions, env_params)

    transition = {
        "obs": state.obs,
        "action": actions,
        "reward": reward,
        "done": done,
        "next_obs": next_obs,
        "info": info,
    }
    new_state = VecState(obs=next_obs, env_state=next_env_state, rng=rng)
    return new_state, transition


def get_dims(env, env_params):
    """Convenience: observation shape and number of discrete actions."""
    obs_shape = env.observation_space(env_params).shape
    num_actions = env.action_space(env_params).n
    return obs_shape, num_actions


# ---------------------------------------------------------------------------
# Achievement / score helpers
# ---------------------------------------------------------------------------

def achievements_from_info(info: dict) -> jnp.ndarray:
    """Pull the per-achievement success flags out of a Craftax ``info`` dict.

    Craftax reports achievements with keys like ``Achievements/collect_wood``.
    Returns a (num_achievements,) array averaged over the batch, or an empty
    array if the keys are absent (older Craktax versions).
    """
    keys = sorted(k for k in info if k.startswith("Achievements"))
    if not keys:
        return jnp.zeros((0,))
    return jnp.stack([jnp.mean(info[k]) for k in keys])


def crafter_score(achievement_rates: jnp.ndarray) -> jnp.ndarray:
    """Crafter/Craftax score = geometric mean of per-achievement success rates.

    score = exp( mean( log(1 + rate*100) ) ) - 1   (rates in [0,1]).
    """
    pct = achievement_rates * 100.0
    return jnp.exp(jnp.mean(jnp.log1p(pct))) - 1.0


# ---------------------------------------------------------------------------
# Benchmark: steps/sec sanity check (Phase 1 "done when")
# ---------------------------------------------------------------------------

def benchmark_steps_per_sec(num_envs: int = 1024, num_steps: int = 200,
                            obs_mode: str = "symbolic", seed: int = 0):
    """Roll a random policy and report env steps/sec. Use to confirm GPU
    vectorization is actually working before training anything."""
    import time

    env, env_params = make_craftax(obs_mode=obs_mode, auto_reset=True)
    _, num_actions = get_dims(env, env_params)
    rng = jax.random.PRNGKey(seed)
    state = reset_vec(env, env_params, rng, num_envs)

    @jax.jit
    def rollout(state):
        def body(state, _):
            state = state._replace(rng=jax.random.fold_in(state.rng, 0))
            act_rng, rng = jax.random.split(state.rng)
            actions = jax.random.randint(act_rng, (num_envs,), 0, num_actions)
            state = state._replace(rng=rng)
            state, _ = step_vec(env, env_params, state, actions)
            return state, None
        state, _ = jax.lax.scan(body, state, None, length=num_steps)
        return state

    state = rollout(state)  # compile
    jax.block_until_ready(state.obs)

    t0 = time.time()
    state = rollout(state)
    jax.block_until_ready(state.obs)
    dt = time.time() - t0

    total = num_envs * num_steps
    sps = total / dt
    print(f"[benchmark] {num_envs} envs x {num_steps} steps = {total:,} "
          f"env-steps in {dt:.3f}s -> {sps:,.0f} steps/sec ({obs_mode})")
    return sps
