"""Actor-critic trained in imagination (Phase 4).

A Dreamer-style behaviour-learning loop that trains entirely inside the
pretrained RSSM's imagination:

  1. Sample real start states (encode short observation contexts -> latent state).
  2. Roll the *actor* forward in imagination using the RSSM prior dynamics.
  3. Predict rewards/continues with the RSSM decoders.
  4. Compute lambda-returns; update critic to predict them and actor to maximise
     them (REINFORCE + entropy, Dreamer-style).

The actor outputs *real* Craftax actions. To drive the latent-dynamics RSSM
(which was pretrained on latent codes), real actions are mapped through the
frozen z->a decoder *inverted*: we instead let the actor emit a latent code and
translate to real actions only at environment-execution time. Here we keep it
simple and have the actor emit over the RSSM's input vocabulary (latent codes),
then translate codes->real actions with the Phase-4 decoder when acting in the
real env. This keeps imagination fully inside the pretrained dynamics.

This module focuses on the imagination training loop and a thin real-env
evaluation wrapper; the heavy lifting (dynamics) is reused from ``rssm.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import distrax

from .rssm import RSSM, RSSMConfig, make_sequence_sampler


class ActorCritic(nn.Module):
    """Actor-critic that reads RSSM features and emits latent-action logits.

    Unlike the PPO baseline (which reads raw observations), this reads the
    pretrained world model's latent feature ``feat = [h, stoch]`` and outputs a
    distribution over the RSSM's *input vocabulary* (the latent codes), not the
    real actions -- so imagination stays inside the learned dynamics.

    Shapes (B = batch; F = feat_dim = deter_dim + stoch_dim = 1536;
    H = hidden = 512; A = num_inputs = codebook K):

        input  feat       : (B, 1536)
        actor:  Dense(H)  : (B, 1536) -> (B, 512)
                Dense(H)  : (B, 512)  -> (B, 512)
                Dense(A)  : (B, 512)  -> (B, A)    logits over latent codes
        critic: Dense(H)  : (B, 1536) -> (B, 512)
                Dense(H)  : (B, 512)  -> (B, 512)
                Dense(1)  : (B, 512)  -> (B, 1) -> squeeze -> (B,)   value
    """
    num_inputs: int          # A: action vocabulary the RSSM consumes (codebook K)
    hidden: int = 512        # H: hidden width

    @nn.compact
    def __call__(self, feat):
        # ----- actor: feat -> logits over latent codes -----
        a = nn.gelu(nn.Dense(self.hidden)(feat))   # (B, 1536) -> (B, 512)
        a = nn.gelu(nn.Dense(self.hidden)(a))      # (B, 512)  -> (B, 512)
        logits = nn.Dense(self.num_inputs)(a)      # (B, 512)  -> (B, A)
        # ----- critic: feat -> scalar value -----
        v = nn.gelu(nn.Dense(self.hidden)(feat))   # (B, 1536) -> (B, 512)
        v = nn.gelu(nn.Dense(self.hidden)(v))      # (B, 512)  -> (B, 512)
        value = jnp.squeeze(nn.Dense(1)(v), -1)    # (B, 1)    -> (B,)
        return distrax.Categorical(logits=logits), value


class AgentConfig(NamedTuple):
    imag_horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4
    entropy_coef: float = 3e-3
    steps: int = 5_000
    batch_size: int = 256
    context: int = 8
    seed: int = 0


def lambda_return(rewards, values, conts, gamma, lam):
    """Dreamer lambda-return over an imagined trajectory (T, B)."""
    def step(carry, inp):
        rew, val, cont = inp
        ret = rew + gamma * cont * ((1 - lam) * val + lam * carry)
        return ret, ret
    last = values[-1]
    _, returns = jax.lax.scan(
        step, last, (rewards[:-1], values[1:], conts[:-1]), reverse=True)
    return returns


def create_agent(num_inputs, feat_dim, cfg: AgentConfig):
    model = ActorCritic(num_inputs)
    key = jax.random.PRNGKey(cfg.seed)
    params = model.init(key, jnp.zeros((1, feat_dim)))
    tx = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(cfg.actor_lr))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


def train_agent_in_imagination(rssm_model: RSSM, rssm_state, rssm_cfg: RSSMConfig,
                               obs, actions, rewards, dones,
                               cfg: AgentConfig | None = None, log_every: int = 200):
    """Train an actor-critic purely in the pretrained RSSM's imagination.

    Returns ``(ac_model, ac_state, history)``. The RSSM is frozen.
    """
    cfg = cfg or AgentConfig()
    feat_dim = rssm_cfg.deter_dim + rssm_cfg.stoch_dim
    ac_model, ac_state = create_agent(rssm_cfg.num_inputs, feat_dim, cfg)

    sampler = make_sequence_sampler(
        obs, actions, rewards, dones, rssm_cfg.num_inputs, cfg.context + 1, cfg.seed)

    def imagine(ac_params, rssm_params, start_h, start_stoch, rng):
        def step(carry, _):
            h, stoch, rng = carry
            feat = jnp.concatenate([h, stoch], axis=-1)
            pi, _ = ac_model.apply(ac_params, feat)
            rng, ar, dr = jax.random.split(rng, 3)
            act = pi.sample(seed=ar)
            act_oh = jax.nn.one_hot(act, rssm_cfg.num_inputs)
            h, stoch, feat_next = rssm_model.apply(
                rssm_params, h, stoch, act_oh, dr, method=rssm_model.imagine_step)
            rew = rssm_model.apply(rssm_params, feat_next, method=rssm_model.decode_reward)
            cont = rssm_model.apply(rssm_params, feat_next, method=rssm_model.decode_cont)
            return (h, stoch, rng), (feat_next, act, rew, cont)

        (_, _, _), (feats, acts, rews, conts) = jax.lax.scan(
            step, (start_h, start_stoch, rng), None, length=cfg.imag_horizon)
        return feats, acts, rews, conts

    @jax.jit
    def update(ac_state, batch, rng):
        # encode context -> start latent state
        r1, r2 = jax.random.split(rng)
        _, _, _, (h, stoch) = rssm_model.apply(
            rssm_state.params, batch["obs"][:cfg.context],
            batch["action"][:cfg.context], r1, method=rssm_model.observe)

        def loss_fn(ac_params):
            feats, acts, rews, conts = imagine(
                ac_params, rssm_state.params, h, stoch, r2)
            pi, values = ac_model.apply(ac_params, feats)
            disc_cont = cfg.gamma * conts
            returns = lambda_return(rews, values, conts, cfg.gamma, cfg.lambda_)

            adv = jax.lax.stop_gradient(returns - values[:-1])
            logp = pi.log_prob(acts)[:-1]
            actor_loss = -(logp * adv).mean() - cfg.entropy_coef * pi.entropy()[:-1].mean()
            critic_loss = jnp.mean((values[:-1] - jax.lax.stop_gradient(returns)) ** 2)
            loss = actor_loss + critic_loss
            return loss, {"loss": loss, "actor_loss": actor_loss,
                          "critic_loss": critic_loss,
                          "imag_reward": rews.mean(),
                          "entropy": pi.entropy().mean()}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ac_state.params)
        ac_state = ac_state.apply_gradients(grads=grads)
        return ac_state, metrics

    rng = jax.random.PRNGKey(cfg.seed)
    history = []
    for s in range(cfg.steps):
        rng, brng = jax.random.split(rng)
        batch = sampler(cfg.batch_size)
        ac_state, metrics = update(ac_state, batch, brng)
        if s % log_every == 0 or s == cfg.steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            m["step"] = s
            history.append(m)
            print(f"[agent] step {s:5d}  loss={m['loss']:.3f}  "
                  f"imag_reward={m['imag_reward']:.4f}  "
                  f"entropy={m['entropy']:.3f}")
    return ac_model, ac_state, history


def evaluate_in_env(ac_model, ac_state, rssm_model, rssm_state, rssm_cfg,
                    code_to_action, num_envs=64, num_steps=512, seed=0):
    """Run the imagination-trained actor in the *real* Craftax env.

    The actor emits latent codes; ``code_to_action`` (length-K int array from the
    Phase-4 decoder) maps each code to a real action. We maintain the RSSM latent
    state online from real observations to pick actions.

    Returns mean achievement score over the rollout.
    """
    from .env_utils import (make_craftax, get_dims, reset_vec, step_vec,
                            achievements_from_info, crafter_score)

    env, env_params = make_craftax("symbolic", auto_reset=True)
    _, num_actions = get_dims(env, env_params)
    code_to_action = jnp.asarray(np.asarray(code_to_action), jnp.int32)

    rng = jax.random.PRNGKey(seed)
    vstate = reset_vec(env, env_params, rng, num_envs)
    h = jnp.zeros((num_envs, rssm_cfg.deter_dim))
    stoch = jnp.zeros((num_envs, rssm_cfg.stoch_dim))

    @jax.jit
    def act(h, stoch, obs, rng):
        # update posterior latent from the current obs (use a zero action-in)
        r1, r2, r3 = jax.random.split(rng, 3)
        # one observe step: treat current stoch/action as carry
        zero_act = jnp.zeros((obs.shape[0], rssm_cfg.num_inputs))
        h2, _, _ = rssm_model.apply(rssm_state.params, h, stoch, zero_act, r1,
                                    method=rssm_model._img_step)
        post_stoch, _ = rssm_model.apply(rssm_state.params, h2, obs, r2,
                                         method=rssm_model._obs_step)
        feat = jnp.concatenate([h2, post_stoch], axis=-1)
        pi, _ = ac_model.apply(ac_state.params, feat)
        code = pi.sample(seed=r3)
        action = code_to_action[code]
        return h2, post_stoch, action

    achievements_acc = []
    for t in range(num_steps):
        rng, ar = jax.random.split(rng)
        h, stoch, action = act(h, stoch, vstate.obs, ar)
        vstate, tr = step_vec(env, env_params, vstate, action)
        ach = achievements_from_info(tr["info"])
        if ach.shape[0] > 0:
            achievements_acc.append(np.asarray(ach))

    if achievements_acc:
        rates = np.mean(achievements_acc, axis=0)
        score = float(crafter_score(jnp.asarray(rates)))
        print(f"[agent] real-env score={score:.2f}  "
              f"mean achievement rate={rates.mean():.3f}")
        return score, rates
    return 0.0, np.zeros(0)
