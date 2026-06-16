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
    """Discrete VQ bottleneck with straight-through gradients.

    Snaps each continuous input vector to its nearest entry in a learned
    codebook of K vectors, returning both the discrete index and the snapped
    vector. The straight-through estimator lets gradients skip the
    non-differentiable argmin so the encoder still trains.

    Shapes (B = batch; K = codebook_size; D = code_dim):
        input  inputs   : (B, D)        continuous embedding from the encoder
        codebook (param): (K, D)        the K learnable code vectors
        dist            : (B, K)        squared L2 distance to every code
        codes           : (B,)          argmin index -> the discrete latent z
        quantized       : (B, D)        codebook[codes], the snapped vector
    """
    codebook_size: int          # K: number of discrete codes (e.g. 32)
    code_dim: int               # D: dimensionality of each code vector (e.g. 64)
    commitment_cost: float = 0.25

    @nn.compact
    def __call__(self, inputs):
        # codebook: (K, D) learnable matrix, one row per discrete code.
        codebook = self.param(
            "codebook",
            nn.initializers.variance_scaling(1.0, "fan_in", "uniform"),
            (self.codebook_size, self.code_dim),
        )
        # Squared L2 distance from each input to each code, via |a-b|^2 =
        # |a|^2 - 2 a.b + |b|^2.  Shapes broadcast to (B, K):
        #   |inputs|^2 : (B, 1)   ;  inputs @ codebook.T : (B, K)  ;  |code|^2 : (1, K)
        dist = (
            jnp.sum(inputs ** 2, axis=1, keepdims=True)        # (B, 1)
            - 2 * inputs @ codebook.T                          # (B, K)
            + jnp.sum(codebook ** 2, axis=1)[None, :]          # (1, K)
        )                                                       # -> (B, K)
        codes = jnp.argmin(dist, axis=1)                  # (B,)  the discrete z_t
        quantized = codebook[codes]                       # (B, D)  gathered code vectors

        # VQ losses (van den Oord 2017): pull codebook toward encoder outputs
        # (codebook_loss) and pull encoder outputs toward codes (commitment_loss).
        codebook_loss = jnp.mean((jax.lax.stop_gradient(inputs) - quantized) ** 2)
        commitment_loss = jnp.mean((inputs - jax.lax.stop_gradient(quantized)) ** 2)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: forward value is `quantized`, but the
        # gradient is routed to `inputs` unchanged (the +stop_gradient(diff)
        # trick makes the two equal in value, identity in gradient).
        quantized_st = inputs + jax.lax.stop_gradient(quantized - inputs)  # (B, D)

        # Perplexity = effective number of codes in use this batch. Near K means
        # healthy usage; near 1 means codebook collapse (a common VQ failure).
        onehot = jax.nn.one_hot(codes, self.codebook_size)   # (B, K)
        probs = jnp.mean(onehot, axis=0)                     # (K,) usage histogram
        perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-10)))  # scalar

        return {
            "quantized": quantized_st,   # (B, D) snapped vector w/ ST gradient
            "codes": codes,              # (B,)   discrete latent action z_t
            "vq_loss": vq_loss,          # scalar
            "perplexity": perplexity,    # scalar diagnostic
        }


# ---------------------------------------------------------------------------
# Conv encoder / decoder
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """Encode a stacked frame pair (H, W, 6) into a flat embedding.

    Four stride-2 conv blocks halve the spatial resolution each time while
    growing the channel count, then a Dense projects the flattened grid to a
    fixed-width embedding.

    Shapes (B = batch; input is the channel-concatenated frame pair, so 6
    channels = two RGB frames; Craftax pixel obs is 63x63):

        input  x          : (B, 63, 63, 6)
        Conv(32,  s2)     : (B, 32, 32, 32)    # ceil(63/2)=32
        Conv(64,  s2)     : (B, 16, 16, 64)
        Conv(128, s2)     : (B,  8,  8, 128)
        Conv(256, s2)     : (B,  4,  4, 256)
        reshape           : (B, 4*4*256) = (B, 4096)
        Dense(embed_dim)  : (B, 256)
    """
    features: tuple = (32, 64, 128, 256)   # output channels per conv block
    embed_dim: int = 256                   # width of the final flat embedding

    @nn.compact
    def __call__(self, x):
        # x starts at (B, 63, 63, 6); each block halves H,W and sets channels=f.
        for f in self.features:
            x = nn.Conv(f, (4, 4), strides=(2, 2), padding="SAME")(x)  # (B, H/2, W/2, f)
            x = nn.LayerNorm()(x)                                       # normalize over last dim
            x = nn.gelu(x)
        x = x.reshape((x.shape[0], -1))      # (B, 4, 4, 256) -> (B, 4096)
        x = nn.Dense(self.embed_dim)(x)      # (B, 4096) -> (B, 256)
        return x


class CNNDecoder(nn.Module):
    """Predict o_{t+1} given o_t (H,W,3) and latent action z (D,).

    The decoder encodes o_t down to a small spatial grid, modulates that grid
    with the latent action z via FiLM (feature-wise affine), then upsamples back
    to a full frame. Conditioning on o_t means the network only has to predict
    the *change* the action caused, not redraw the whole scene.

    Shapes (B = batch; out_shape = (63, 63, 3); D = code_dim = 64):

        input  o_t            : (B, 63, 63, 3)
        input  z              : (B, 64)
        -- downsample o_t --
        Conv(32,  s2)         : (B, 32, 32, 32)
        Conv(64,  s2)         : (B, 16, 16, 64)
        Conv(128, s2)         : (B,  8,  8, 128)
        Conv(256, s2)         : (B,  4,  4, 256)
        -- FiLM modulation from z --
        Dense(256) -> gamma   : (B, 256) -> broadcast (B, 1, 1, 256)
        Dense(256) -> beta    : (B, 256) -> broadcast (B, 1, 1, 256)
        x = x*(1+gamma)+beta  : (B, 4, 4, 256)
        -- upsample (ConvTranspose doubles H,W each block) --
        ConvT(256, s2)        : (B,  8,  8, 256)
        ConvT(128, s2)        : (B, 16, 16, 128)
        ConvT(64,  s2)        : (B, 32, 32, 64)
        ConvT(32,  s2)        : (B, 64, 64, 32)
        Conv(3)               : (B, 64, 64, 3)
        resize -> out_shape   : (B, 63, 63, 3)   (sigmoid -> pixels in [0,1])
    """
    out_shape: tuple             # (H, W, C) of the predicted frame, e.g. (63,63,3)
    features: tuple = (256, 128, 64, 32)   # upsampling channels per block

    @nn.compact
    def __call__(self, o_t, z):
        H, W, C = self.out_shape
        # ----- downsample o_t to a (B, 4, 4, 256) latent grid -----
        x = o_t                               # (B, 63, 63, 3)
        for f in (32, 64, 128, 256):
            x = nn.Conv(f, (4, 4), strides=(2, 2), padding="SAME")(x)  # halves H,W
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        # ----- FiLM: scale+shift the grid by an affine fn of the latent z -----
        # gamma/beta: Dense maps z (B, D) -> (B, 256), reshaped to (B,1,1,256)
        # so they broadcast across the 4x4 spatial grid.
        gamma = nn.Dense(x.shape[-1])(z)[:, None, None, :]   # (B, 1, 1, 256)
        beta = nn.Dense(x.shape[-1])(z)[:, None, None, :]    # (B, 1, 1, 256)
        x = x * (1 + gamma) + beta                           # (B, 4, 4, 256)

        # ----- upsample back to full resolution -----
        for f in self.features:
            x = nn.ConvTranspose(f, (4, 4), strides=(2, 2), padding="SAME")(x)  # doubles H,W
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)
        x = nn.Conv(C, (3, 3), padding="SAME")(x)            # (B, 64, 64, 3)
        # ConvTranspose lands on 64x64; resize to the exact obs shape (63x63).
        x = jax.image.resize(x, (x.shape[0], H, W, C), method="bilinear")  # (B, 63, 63, 3)
        return nn.sigmoid(x)                                 # pixels in [0,1]


# ---------------------------------------------------------------------------
# Full latent-action model
# ---------------------------------------------------------------------------

class LatentActionModel(nn.Module):
    """Full inverse+forward dynamics model (the Phase-2 novel piece).

    Data flow (B = batch; frames are 63x63x3; D = code_dim; K = codebook_size):

        o_t, o_tp1                  : each (B, 63, 63, 3)
        concat on channels -> pair  : (B, 63, 63, 6)
        CNNEncoder(pair)            : (B, 256)          inverse-dynamics embedding
        Dense(D)                    : (B, 64)
        VectorQuantizer             : code z_t (B,) + quantized (B, 64)
        CNNDecoder(o_t, quantized)  : (B, 63, 63, 3)    forward prediction of o_tp1
    """
    out_shape: tuple             # (H, W, C) of a single frame, e.g. (63,63,3)
    codebook_size: int = 32      # K: number of discrete latent actions
    code_dim: int = 64           # D: width of the pre-quantization embedding
    commitment_cost: float = 0.25

    @nn.compact
    def __call__(self, o_t, o_tp1):
        pair = jnp.concatenate([o_t, o_tp1], axis=-1)     # (B, 63, 63, 6)
        h = CNNEncoder()(pair)                            # (B, 256)
        h = nn.Dense(self.code_dim)(h)                    # (B, 64) project to code dim
        vq = VectorQuantizer(self.codebook_size, self.code_dim,
                             self.commitment_cost)(h)      # codes (B,), quantized (B, 64)
        recon = CNNDecoder(self.out_shape)(o_t, vq["quantized"])  # (B, 63, 63, 3)
        return {"recon": recon, **vq}

    def encode(self, o_t, o_tp1):
        """Return only the discrete code z_t (B,) for downstream phases.

        Same encoder+VQ path as __call__ but skips the decoder, so it's cheap to
        run over the whole dataset when extracting latent actions.
        """
        pair = jnp.concatenate([o_t, o_tp1], axis=-1)     # (B, 63, 63, 6)
        h = CNNEncoder()(pair)                            # (B, 256)
        h = nn.Dense(self.code_dim)(h)                    # (B, 64)
        vq = VectorQuantizer(self.codebook_size, self.code_dim,
                             self.commitment_cost)(h)
        return vq["codes"]                                # (B,) discrete latent actions


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
