"""Latent Action Model -- LAPA/Genie-style VQ-VAE on frame pairs (Phase 2).

Architecture (inverse-dynamics + forward-dynamics, à la Genie/LAPA):

    o_t, o_{t+1}  --[CNN encoder]-->  h  --[VQ]-->  z_t  (discrete latent action)
    o_t, z_t      --[CNN decoder]-->  ô_{t+1}        (forward prediction)

The model learns *what changed* between two consecutive frames and compresses it
into one of K discrete codes. Because Craftax-classic has 17 actions, a codebook
of K~=16-32 should -- if learning works -- recover an action-like quantization
*without ever seeing the action labels*. Validation (NMI / confusion matrix vs.
the true actions) lives in ``eval.py`` and is called from the notebook.

Pure Flax/JAX. The VQ layer uses the straight-through estimator with a
commitment loss (van den Oord et al., 2017).

References
----------
LAPA: https://arxiv.org/abs/2410.11758
VQ-VAE: https://arxiv.org/abs/1711.00937
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


# ---------------------------------------------------------------------------
# Vector-quantization bottleneck
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Discrete VQ bottleneck with straight-through gradients."""
    codebook_size: int          # K
    code_dim: int               # D
    commitment_cost: float = 0.25

    @nn.compact
    def __call__(self, inputs):
        # inputs: (B, D)
        codebook = self.param(
            "codebook",
            nn.initializers.variance_scaling(1.0, "fan_in", "uniform"),
            (self.codebook_size, self.code_dim),
        )
        # squared L2 distance to each code
        dist = (
            jnp.sum(inputs ** 2, axis=1, keepdims=True)
            - 2 * inputs @ codebook.T
            + jnp.sum(codebook ** 2, axis=1)[None, :]
        )
        codes = jnp.argmin(dist, axis=1)                  # (B,)
        quantized = codebook[codes]                       # (B, D)

        # losses
        codebook_loss = jnp.mean((jax.lax.stop_gradient(inputs) - quantized) ** 2)
        commitment_loss = jnp.mean((inputs - jax.lax.stop_gradient(quantized)) ** 2)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # straight-through: gradient flows to inputs unchanged
        quantized_st = inputs + jax.lax.stop_gradient(quantized - inputs)

        # codebook usage perplexity (diagnostic for collapse)
        onehot = jax.nn.one_hot(codes, self.codebook_size)
        probs = jnp.mean(onehot, axis=0)
        perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-10)))

        return {
            "quantized": quantized_st,
            "codes": codes,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
        }


# ---------------------------------------------------------------------------
# Conv encoder / decoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Encode a stacked frame pair (H, W, 6) into a flat embedding."""
    features: tuple = (32, 64, 128, 256)
    embed_dim: int = 256

    @nn.compact
    def __call__(self, x):
        for f in self.features:
            x = nn.Conv(f, (4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.embed_dim)(x)
        return x


class CNNDecoder(nn.Module):
    """Predict o_{t+1} given o_t (H,W,3) and latent action z (D,).

    Conditioning is by FiLM-style modulation injected at the bottleneck plus
    concatenation of o_t so the decoder mostly copies and edits the frame.
    """
    out_shape: tuple             # (H, W, 3)
    features: tuple = (256, 128, 64, 32)

    @nn.compact
    def __call__(self, o_t, z):
        H, W, C = self.out_shape
        # encode o_t to a small spatial grid
        x = o_t
        for f in (32, 64, 128, 256):
            x = nn.Conv(f, (4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        # x: (B, h, w, 256); inject z via FiLM
        gamma = nn.Dense(x.shape[-1])(z)[:, None, None, :]
        beta = nn.Dense(x.shape[-1])(z)[:, None, None, :]
        x = x * (1 + gamma) + beta

        for f in self.features:
            x = nn.ConvTranspose(f, (4, 4), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        x = nn.Conv(C, (3, 3), padding="SAME")(x)
        # crop/resize to exact out shape
        x = jax.image.resize(x, (x.shape[0], H, W, C), method="bilinear")
        return nn.sigmoid(x)


# ---------------------------------------------------------------------------
# Full latent-action model
# ---------------------------------------------------------------------------

class LatentActionModel(nn.Module):
    out_shape: tuple             # (H, W, 3)
    codebook_size: int = 32
    code_dim: int = 64
    commitment_cost: float = 0.25

    @nn.compact
    def __call__(self, o_t, o_tp1):
        pair = jnp.concatenate([o_t, o_tp1], axis=-1)     # (B,H,W,6)
        h = CNNEncoder()(pair)
        h = nn.Dense(self.code_dim)(h)
        vq = VectorQuantizer(self.codebook_size, self.code_dim,
                             self.commitment_cost)(h)
        recon = CNNDecoder(self.out_shape)(o_t, vq["quantized"])
        return {"recon": recon, **vq}

    def encode(self, o_t, o_tp1):
        """Return only the discrete code z_t (for downstream phases)."""
        pair = jnp.concatenate([o_t, o_tp1], axis=-1)
        h = CNNEncoder()(pair)
        h = nn.Dense(self.code_dim)(h)
        vq = VectorQuantizer(self.codebook_size, self.code_dim,
                             self.commitment_cost)(h)
        return vq["codes"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class LAMConfig(NamedTuple):
    codebook_size: int = 32
    code_dim: int = 64
    commitment_cost: float = 0.25
    lr: float = 3e-4
    batch_size: int = 256
    steps: int = 20_000
    seed: int = 0


def _to_float(frames_uint8):
    return frames_uint8.astype(jnp.float32) / 255.0


def create_lam_state(out_shape, config: LAMConfig):
    model = LatentActionModel(
        out_shape=out_shape,
        codebook_size=config.codebook_size,
        code_dim=config.code_dim,
        commitment_cost=config.commitment_cost,
    )
    rng = jax.random.PRNGKey(config.seed)
    dummy = jnp.zeros((1, *out_shape))
    params = model.init(rng, dummy, dummy)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config.lr))
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return model, state


def lam_loss_fn(params, apply_fn, o_t, o_tp1, commitment_cost):
    out = apply_fn(params, o_t, o_tp1)
    recon_loss = jnp.mean((out["recon"] - o_tp1) ** 2)
    loss = recon_loss + out["vq_loss"]
    metrics = {
        "loss": loss,
        "recon_loss": recon_loss,
        "vq_loss": out["vq_loss"],
        "perplexity": out["perplexity"],
    }
    return loss, metrics


@jax.jit
def lam_train_step(state, o_t, o_tp1):
    o_t, o_tp1 = _to_float(o_t), _to_float(o_tp1)

    def loss_fn(params):
        return lam_loss_fn(params, state.apply_fn, o_t, o_tp1, 0.25)

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def train_lam(pairs: dict, out_shape, config: LAMConfig | None = None,
              log_every: int = 500):
    """Train the latent-action model on (o_t, o_tp1) pairs.

    ``pairs`` is the dict from ``collect.make_pair_dataset``. Returns
    ``(model, state, history)``.
    """
    import numpy as np
    config = config or LAMConfig()
    model, state = create_lam_state(out_shape, config)

    o_t_all = pairs["o_t"]
    o_tp1_all = pairs["o_tp1"]
    n = len(o_t_all)
    rng = np.random.default_rng(config.seed)
    history = []

    for step in range(config.steps):
        idx = rng.integers(0, n, size=config.batch_size)
        state, metrics = lam_train_step(
            state, jnp.asarray(o_t_all[idx]), jnp.asarray(o_tp1_all[idx]))
        if step % log_every == 0 or step == config.steps - 1:
            m = {k: float(v) for k, v in metrics.items()}
            m["step"] = step
            history.append(m)
            print(f"[lam] step {step:6d}  loss={m['loss']:.4f}  "
                  f"recon={m['recon_loss']:.4f}  vq={m['vq_loss']:.4f}  "
                  f"perplexity={m['perplexity']:.2f}/{config.codebook_size}")
    return model, state, history


def encode_codes(model, state, o_t, o_tp1, batch_size: int = 1024):
    """Run the encoder over a (possibly large) array of pairs, returning the
    discrete code per pair. Used for validation and Phase 3 inputs."""
    import numpy as np

    @jax.jit
    def enc(params, a, b):
        return model.apply(params, _to_float(a), _to_float(b), method=model.encode)

    codes = []
    for i in range(0, len(o_t), batch_size):
        c = enc(state.params, jnp.asarray(o_t[i:i + batch_size]),
                jnp.asarray(o_tp1[i:i + batch_size]))
        codes.append(np.asarray(c))
    return np.concatenate(codes)
