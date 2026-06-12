"""Baseline PPO agent for Craftax-classic (Phase 1).

A compact, fully-jitted PPO implementation in the PureJaxRL style: the whole
training loop is a single ``jax.lax.scan`` so it compiles to one XLA program and
runs entirely on-device. Operates on the *symbolic* observation (flat vector)
for speed; this is the baseline the latent-action pipeline is compared against.

The same trained policy is reused in ``collect.py`` to gather a labelled
trajectory dataset for Phases 2-4.
"""

from __future__ import annotations

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import distrax

from .env_utils import make_craftax, get_dims, achievements_from_info


# ---------------------------------------------------------------------------
# Network: shared MLP torso -> categorical actor + scalar critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    num_actions: int
    hidden: int = 512

    @nn.compact
    def __call__(self, x):
        act = nn.relu(nn.Dense(self.hidden)(x))
        act = nn.relu(nn.Dense(self.hidden)(act))
        logits = nn.Dense(self.num_actions,
                          kernel_init=nn.initializers.orthogonal(0.01))(act)
        pi = distrax.Categorical(logits=logits)

        val = nn.relu(nn.Dense(self.hidden)(x))
        val = nn.relu(nn.Dense(self.hidden)(val))
        value = nn.Dense(1)(val)
        return pi, jnp.squeeze(value, -1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class PPOConfig(NamedTuple):
    total_timesteps: int = 5_000_000
    num_envs: int = 1024
    num_steps: int = 64          # rollout horizon per update
    num_minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    anneal_lr: bool = True
    seed: int = 0


def make_train(config: PPOConfig):
    """Build a fully-jitted PPO ``train`` function. Returns ``train(rng)`` which
    yields a dict with the final ``train_state`` and logged metrics."""
    env, env_params = make_craftax("symbolic", auto_reset=True)
    obs_shape, num_actions = get_dims(env, env_params)

    num_updates = config.total_timesteps // (config.num_envs * config.num_steps)
    minibatch_size = config.num_envs * config.num_steps // config.num_minibatches

    def lr_schedule(count):
        frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / num_updates
        return config.lr * frac

    def train(rng):
        net = ActorCritic(num_actions=num_actions)
        rng, init_rng = jax.random.split(rng)
        dummy = jnp.zeros((1, *obs_shape))
        params = net.init(init_rng, dummy)

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(lr_schedule if config.anneal_lr else config.lr, eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=net.apply, params=params, tx=tx)

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)

        # --- one full PPO update ------------------------------------------
        def update_step(runner_state, _):
            def env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state
                rng, a_rng = jax.random.split(rng)
                pi, value = net.apply(train_state.params, last_obs)
                action = pi.sample(seed=a_rng)
                log_prob = pi.log_prob(action)

                rng, s_rng = jax.random.split(rng)
                s_rngs = jax.random.split(s_rng, config.num_envs)
                obsv, env_state_n, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(s_rngs, env_state, action, env_params)

                transition = Transition(done, action, value, reward, log_prob, last_obs)
                return (train_state, env_state_n, obsv, rng), (transition, info)

            runner_state, (traj, info) = jax.lax.scan(
                env_step, runner_state, None, config.num_steps)

            # --- GAE -------------------------------------------------------
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = net.apply(train_state.params, last_obs)

            def gae_step(carry, t):
                gae, next_val = carry
                delta = t.reward + config.gamma * next_val * (1 - t.done) - t.value
                gae = delta + config.gamma * config.gae_lambda * (1 - t.done) * gae
                return (gae, t.value), gae

            _, advantages = jax.lax.scan(
                gae_step, (jnp.zeros_like(last_val), last_val), traj, reverse=True)
            targets = advantages + traj.value

            # --- epochs of minibatch SGD ----------------------------------
            def epoch(carry, _):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                batch_size = config.num_envs * config.num_steps
                perm = jax.random.permutation(perm_rng, batch_size)

                flat = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj)
                adv = advantages.reshape(batch_size)
                tgt = targets.reshape(batch_size)
                flat, adv, tgt = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, perm, axis=0), (flat, adv, tgt))

                def minibatch(train_state, mb):
                    mb_traj, mb_adv, mb_tgt = mb

                    def loss_fn(params):
                        pi, value = net.apply(params, mb_traj.obs)
                        log_prob = pi.log_prob(mb_traj.action)
                        ratio = jnp.exp(log_prob - mb_traj.log_prob)
                        a = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        l1 = ratio * a
                        l2 = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * a
                        actor_loss = -jnp.minimum(l1, l2).mean()

                        v_clip = mb_traj.value + jnp.clip(
                            value - mb_traj.value, -config.clip_eps, config.clip_eps)
                        v_loss = jnp.maximum((value - mb_tgt) ** 2,
                                             (v_clip - mb_tgt) ** 2).mean()
                        entropy = pi.entropy().mean()
                        total = actor_loss + config.vf_coef * 0.5 * v_loss \
                            - config.ent_coef * entropy
                        return total, (actor_loss, v_loss, entropy)

                    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, *aux)

                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((config.num_minibatches, minibatch_size)
                                        + x.shape[1:]),
                    (flat, adv, tgt))
                train_state, losses = jax.lax.scan(minibatch, train_state, minibatches)
                return (train_state, rng), losses

            (train_state, rng), losses = jax.lax.scan(
                epoch, (train_state, rng), None, config.update_epochs)

            # log mean reward + achievement score this update
            metric = {
                "reward": traj.reward.mean(),
                "loss": losses[0].mean(),
                "entropy": losses[3].mean(),
            }
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, run_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, run_rng)
        runner_state, metrics = jax.lax.scan(
            update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train, (env, env_params, num_actions, obs_shape)


def train_ppo(config: PPOConfig | None = None):
    """Train PPO and return (train_state, metrics, env_bundle)."""
    config = config or PPOConfig()
    train, env_bundle = make_train(config)
    out = jax.jit(train)(jax.random.PRNGKey(config.seed))
    train_state = out["runner_state"][0]
    return train_state, out["metrics"], env_bundle
