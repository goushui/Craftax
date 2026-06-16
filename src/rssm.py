"""Dreamer-style Recurrent State-Space Model (Phase 3).

A simplified DreamerV3 RSSM that learns the dynamics of Craftax from sequences of
observations and *actions* -- where "action" is either the learned discrete
latent action z_t (the novel pipeline) or the true action a_t (the sanity-check
baseline). Training on both and comparing multi-step rollout error tells us how
much information the latent-action quantization loses.

State factorization (Dreamer):
    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])         deterministic recurrent state
    prior  p(ẑ_t | h_t)                            categorical, used in imagination
    post   q(z_t | h_t, o_t)                        categorical, used in training

We use the symbolic Craftax observation (flat vector) here for speed -- the
RSSM's job is dynamics, and the pixel-level work is done by the Phase-2 LAM.

Losses: observation reconstruction + reward + done prediction + KL(post||prior)
with free bits and KL balancing.

References
----------
DreamerV3: https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * (jnp.expm1(jnp.abs(x)))


# ---------------------------------------------------------------------------
# Categorical latent helpers (discrete stochastic state, Dreamer-style)
# ---------------------------------------------------------------------------

def categorical_sample(logits, rng):
    return jax.random.categorical(rng, logits, axis=-1)


def onehot_st(logits, rng):
    """Sample a one-hot with straight-through gradients."""
    sample = jax.random.categorical(rng, logits, axis=-1)
    onehot = jax.nn.one_hot(sample, logits.shape[-1])
    probs = jax.nn.softmax(logits, axis=-1)
    return onehot + (probs - jax.lax.stop_gradient(probs))


# ---------------------------------------------------------------------------
# RSSM core
# ---------------------------------------------------------------------------

class RSSMConfig(NamedTuple):
    deter_dim: int = 512          # GRU hidden size (h)
    stoch_categories: int = 32    # number of categorical variables
    stoch_classes: int = 32       # classes per variable (z is 32x32 one-hot)
    hidden: int = 512
    obs_dim: int = 8268           # Craftax-classic symbolic obs size (auto-set)
    num_inputs: int = 32          # action-input cardinality (codebook K or num_actions)
    free_bits: float = 1.0
    kl_scale: float = 1.0
    kl_balance: float = 0.8
    lr: float = 3e-4
    seq_len: int = 32
    batch_size: int = 32
    steps: int = 20_000
    seed: int = 0

    @property
    def stoch_dim(self):
        return self.stoch_categories * self.stoch_classes


class RSSM(nn.Module):
    """Recurrent State-Space Model.

    The latent state has two parts (Dreamer factorization):
      * h     -- deterministic GRU hidden state, size ``deter_dim`` (512)
      * stoch -- stochastic state, ``stoch_categories`` x ``stoch_classes``
                 one-hot variables flattened to ``stoch_dim`` = 32*32 = 1024.
    The decoders read the concatenated feature ``[h, stoch]`` of size
    ``deter_dim + stoch_dim`` = 1536.

    Dimension glossary used in the layer comments below (defaults in parens):
      A   = num_inputs   action vocabulary (codebook K or num_actions)
      Hd  = hidden       MLP width (512)
      De  = deter_dim    GRU state size (512)
      S   = stoch_dim    flattened stochastic size (1024 = 32*32)
      O   = obs_dim      symbolic observation size (~8268)
    """
    cfg: RSSMConfig

    def setup(self):
        c = self.cfg
        # Layer  (in -> out):
        self.act_embed = nn.Dense(c.hidden)      # A      -> Hd   embed the action
        self.gru_in = nn.Dense(c.hidden)         # S+Hd   -> Hd   pre-GRU mixing
        self.gru = nn.GRUCell(features=c.deter_dim)  # (De, Hd) -> De  recurrent core
        self.prior_mlp = nn.Dense(c.hidden)      # De     -> Hd
        self.prior_logits = nn.Dense(c.stoch_dim)  # Hd   -> S    prior over next stoch
        self.obs_in = nn.Dense(c.hidden)         # O      -> Hd   embed the observation
        self.post_mlp = nn.Dense(c.hidden)       # De+Hd  -> Hd
        self.post_logits = nn.Dense(c.stoch_dim)  # Hd    -> S    posterior over stoch
        # Decoder heads read feat = [h, stoch] of size De+S = 1536.
        self.obs_dec = nn.Sequential([nn.Dense(c.hidden), nn.gelu,   # 1536 -> Hd
                                      nn.Dense(c.hidden), nn.gelu,    # Hd   -> Hd
                                      nn.Dense(c.obs_dim)])           # Hd   -> O  (reconstruct obs)
        self.rew_dec = nn.Sequential([nn.Dense(c.hidden), nn.gelu,   # 1536 -> Hd
                                      nn.Dense(1)])                   # Hd   -> 1  (reward)
        self.cont_dec = nn.Sequential([nn.Dense(c.hidden), nn.gelu,  # 1536 -> Hd
                                       nn.Dense(1)])                  # Hd   -> 1  (continue prob)

    # -- one step of the recurrent prior -------------------------------------
    def _img_step(self, h, stoch, action_onehot, rng):
        """Advance the deterministic state and sample the *prior* stoch (the
        model's guess of the next stochastic state without seeing the obs).

        Shapes (B = batch):
            h             : (B, De=512)
            stoch         : (B, S=1024)
            action_onehot : (B, A)
        """
        c = self.cfg
        a = self.act_embed(action_onehot)         # (B, A)    -> (B, Hd=512)
        x = jnp.concatenate([stoch, a], axis=-1)  # (B, S+Hd = 1536)
        x = nn.gelu(self.gru_in(x))               # (B, 1536) -> (B, Hd=512)
        h, _ = self.gru(h, x)                     # GRUCell((B,De),(B,Hd)) -> (B, De)
        # prior over next stochastic state, as logits reshaped to per-variable
        p = nn.gelu(self.prior_mlp(h))            # (B, De)   -> (B, Hd)
        prior_logits = self.prior_logits(p).reshape(  # (B, Hd) -> (B, S) ->
            (-1, c.stoch_categories, c.stoch_classes))  # (B, 32, 32)
        rng, srng = jax.random.split(rng)
        prior_stoch = onehot_st(prior_logits, srng).reshape((-1, c.stoch_dim))  # (B, S)
        return h, prior_stoch, prior_logits

    def _obs_step(self, h, obs, rng):
        """Sample the *posterior* stoch from h and the actual observation.

        Shapes:
            h   : (B, De=512)
            obs : (B, O~=8268)
        """
        c = self.cfg
        e = nn.gelu(self.obs_in(symlog(obs)))     # (B, O)    -> (B, Hd=512)
        x = jnp.concatenate([h, e], axis=-1)      # (B, De+Hd = 1024)
        x = nn.gelu(self.post_mlp(x))             # (B, 1024) -> (B, Hd=512)
        post_logits = self.post_logits(x).reshape(  # (B, Hd) -> (B, S) ->
            (-1, c.stoch_categories, c.stoch_classes))  # (B, 32, 32)
        rng, srng = jax.random.split(rng)
        post_stoch = onehot_st(post_logits, srng).reshape((-1, c.stoch_dim))  # (B, S)
        return post_stoch, post_logits

    def initial(self, batch):
        c = self.cfg
        return {
            "h": jnp.zeros((batch, c.deter_dim)),
            "stoch": jnp.zeros((batch, c.stoch_dim)),
        }

    # -- full sequence observe (training) ------------------------------------
    def observe(self, obs_seq, action_seq, rng):
        """obs_seq: (T, B, obs_dim); action_seq: (T, B, num_inputs) one-hot.

        Returns per-step prior/post logits and the latent features used by the
        decoders. action_seq[t] is the action that *leads into* obs[t].
        """
        c = self.cfg
        T, B = obs_seq.shape[0], obs_seq.shape[1]
        state = self.initial(B)

        def step(carry, inp):
            h, stoch, rng = carry
            obs_t, act_t = inp
            rng, r1, r2 = jax.random.split(rng, 3)
            h, _, prior_logits = self._img_step(h, stoch, act_t, r1)
            post_stoch, post_logits = self._obs_step(h, obs_t, r2)
            feat = jnp.concatenate([h, post_stoch], axis=-1)
            return (h, post_stoch, rng), (prior_logits, post_logits, feat)

        (h, stoch, _), (prior_logits, post_logits, feat) = jax.lax.scan(
            step, (state["h"], state["stoch"], rng), (obs_seq, action_seq))
        return prior_logits, post_logits, feat, (h, stoch)

    # -- imagination rollout from a latent state (Phase 4) -------------------
    def imagine_step(self, h, stoch, action_onehot, rng):
        h, prior_stoch, _ = self._img_step(h, stoch, action_onehot, rng)
        feat = jnp.concatenate([h, prior_stoch], axis=-1)
        return h, prior_stoch, feat

    def decode_obs(self, feat):
        return self.obs_dec(feat)

    def decode_reward(self, feat):
        return symexp(jnp.squeeze(self.rew_dec(feat), -1))

    def decode_cont(self, feat):
        return jax.nn.sigmoid(jnp.squeeze(self.cont_dec(feat), -1))


# ---------------------------------------------------------------------------
# Loss + training
# ---------------------------------------------------------------------------

def _kl_categorical(post_logits, prior_logits, free_bits, balance):
    """KL(post || prior) over categorical latents with KL balancing + free bits."""
    post = jax.nn.softmax(post_logits, -1)
    logpost = jax.nn.log_softmax(post_logits, -1)
    logprior = jax.nn.log_softmax(prior_logits, -1)

    def kl(lp, lq):
        p = jax.nn.softmax(lp, -1)
        return jnp.sum(p * (jax.nn.log_softmax(lp, -1) - lq), axis=-1)

    kl_lhs = kl(jax.lax.stop_gradient(post_logits), logprior)   # train prior
    kl_rhs = kl(post_logits, jax.lax.stop_gradient(logprior))   # train post
    value = balance * kl_lhs + (1 - balance) * kl_rhs
    value = jnp.sum(value, axis=-1)        # sum over categorical vars
    value = jnp.maximum(value, free_bits)
    return jnp.mean(value)


def rssm_loss_fn(params, model: RSSM, batch, rng):
    cfg = model.cfg
    obs = symlog(batch["obs"])                       # not used directly; symlog inside
    prior_logits, post_logits, feat, _ = model.apply(
        params, batch["obs"], batch["action"], rng, method=model.observe)

    obs_pred = model.apply(params, feat, method=model.decode_obs)
    rew_pred = model.apply(params, feat, method=model.decode_reward)
    cont_pred = model.apply(params, feat, method=model.decode_cont)

    obs_loss = jnp.mean((obs_pred - symlog(batch["obs"])) ** 2)
    rew_loss = jnp.mean((rew_pred - batch["reward"]) ** 2)
    cont_target = 1.0 - batch["done"].astype(jnp.float32)
    cont_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
        jnp.log(cont_pred / (1 - cont_pred) + 1e-8), cont_target))

    kl_loss = _kl_categorical(post_logits, prior_logits, cfg.free_bits, cfg.kl_balance)

    loss = obs_loss + rew_loss + cont_loss + cfg.kl_scale * kl_loss
    metrics = {"loss": loss, "obs_loss": obs_loss, "rew_loss": rew_loss,
               "cont_loss": cont_loss, "kl": kl_loss}
    return loss, metrics


def create_rssm_state(cfg: RSSMConfig):
    model = RSSM(cfg)
    rng = jax.random.PRNGKey(cfg.seed)
    obs = jnp.zeros((cfg.seq_len, cfg.batch_size, cfg.obs_dim))
    act = jnp.zeros((cfg.seq_len, cfg.batch_size, cfg.num_inputs))
    params = model.init(rng, obs, act, rng, method=model.observe)
    tx = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(cfg.lr))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


def make_sequence_sampler(obs, actions, rewards, dones, num_inputs, seq_len, seed=0):
    """Build a sampler yielding (T,B,...) batches of contiguous sequences.

    ``actions`` are integer ids (true actions or latent codes); they are
    one-hot encoded to width ``num_inputs``. Sequences are sampled to avoid
    crossing the end of the flat buffer; episode boundaries inside a sequence
    are fine because ``done`` is part of the supervision.
    """
    import numpy as np
    obs = np.asarray(obs, np.float32)
    actions = np.asarray(actions, np.int32)
    rewards = np.asarray(rewards, np.float32)
    dones = np.asarray(dones, bool)
    N = len(obs)
    rng = np.random.default_rng(seed)

    def sample(batch_size):
        starts = rng.integers(0, N - seq_len - 1, size=batch_size)
        idx = starts[:, None] + np.arange(seq_len)[None, :]      # (B, T)
        idx = idx.T                                               # (T, B)
        a = actions[idx]
        a_onehot = np.eye(num_inputs, dtype=np.float32)[a]
        return {
            "obs": jnp.asarray(obs[idx]),
            "action": jnp.asarray(a_onehot),
            "reward": jnp.asarray(rewards[idx]),
            "done": jnp.asarray(dones[idx]),
        }
    return sample


def train_rssm(obs, actions, rewards, dones, cfg: RSSMConfig,
               log_every: int = 500):
    """Train an RSSM. ``actions`` is the integer action stream -- pass the true
    actions for the baseline run and the latent codes for the novel run."""
    model, state = create_rssm_state(cfg)
    sampler = make_sequence_sampler(
        obs, actions, rewards, dones, cfg.num_inputs, cfg.seq_len, cfg.seed)

    @jax.jit
    def step(state, batch, rng):
        def loss_fn(p):
            return rssm_loss_fn(p, model, batch, rng)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    rng = jax.random.PRNGKey(cfg.seed)
    history = []
    for s in range(cfg.steps):
        rng, brng = jax.random.split(rng)
        batch = sampler(cfg.batch_size)
        state, metrics = step(state, batch, brng)
        if s % log_every == 0 or s == cfg.steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            m["step"] = s
            history.append(m)
            print(f"[rssm] step {s:6d}  loss={m['loss']:.3f}  "
                  f"obs={m['obs_loss']:.3f}  rew={m['rew_loss']:.4f}  "
                  f"kl={m['kl']:.3f}")
    return model, state, history


def rollout_error(model, state, obs, actions, rewards, dones, cfg: RSSMConfig,
                  horizon: int = 15, num_seqs: int = 256, seed: int = 123):
    """Multi-step open-loop rollout error: observe a context, then imagine
    ``horizon`` steps using only the action stream, and measure obs MSE vs.
    ground truth. This is the headline Phase-3 comparison metric (latent vs.
    true action RSSM)."""
    import numpy as np
    sampler = make_sequence_sampler(
        obs, actions, rewards, dones, cfg.num_inputs, horizon + 8, seed)
    batch = sampler(num_seqs)

    context = 8

    def run(params, batch, rng):
        # observe the context to get a latent state
        ctx_obs = batch["obs"][:context]
        ctx_act = batch["action"][:context]
        _, _, _, (h, stoch) = model.apply(
            params, ctx_obs, ctx_act, rng, method=model.observe)

        # imagine forward using the *true* action stream but no obs
        def step(carry, act_t):
            h, stoch, rng = carry
            rng, r = jax.random.split(rng)
            h, stoch, feat = model.apply(
                params, h, stoch, act_t, r, method=model.imagine_step)
            obs_pred = model.apply(params, feat, method=model.decode_obs)
            return (h, stoch, rng), obs_pred

        fut_act = batch["action"][context:context + horizon]
        (_, _, _), preds = jax.lax.scan(step, (h, stoch, rng), fut_act)
        return preds

    rng = jax.random.PRNGKey(seed)
    preds = jax.jit(run)(state.params, batch, rng)        # (horizon, B, obs)
    target = symlog(batch["obs"][context:context + horizon])
    mse_per_step = jnp.mean((preds - target) ** 2, axis=(1, 2))
    print(f"[rssm] open-loop rollout MSE per step (h={horizon}): "
          f"{np.round(np.asarray(mse_per_step), 4)}")
    return np.asarray(mse_per_step)
