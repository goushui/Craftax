"""Latent-action -> real-action decoder (Phase 4).

A small MLP that maps a learned latent code z_t (one-hot over the codebook) to a
distribution over the 17 real Craftax actions. Trained on a *small labelled
subset* of the trajectory data; the headline experiment sweeps the label budget
(1%, 5%, 25%, 100%) and plots decoding accuracy vs. budget -- the core
label-efficiency result called out in the plan's fallback notes.

Because z_t is discrete, the "decoder" can be as simple as a learned lookup
table (codebook_size x num_actions), but we keep a tiny MLP so it also works if
you later feed continuous latent features.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


class ActionDecoder(nn.Module):
    """Tiny MLP mapping a one-hot latent code to real-action logits.

    Input is the latent action z as a one-hot vector over the codebook, output
    is one logit per real Craftax action.

    Shapes (B = batch; K = codebook_size; H = hidden = 128; A = num_actions = 17):

        input  z_onehot   : (B, K)
        Dense(H) + relu   : (B, K) -> (B, 128)
        Dense(A)          : (B, 128) -> (B, A)   logits over real actions
    """
    num_actions: int          # A: real Craftax action count (17)
    codebook_size: int        # K: latent codebook size (input width)
    hidden: int = 128         # H

    @nn.compact
    def __call__(self, z_onehot):
        x = nn.relu(nn.Dense(self.hidden)(z_onehot))  # (B, K) -> (B, 128)
        return nn.Dense(self.num_actions)(x)          # (B, 128) -> (B, A)


class DecoderConfig(NamedTuple):
    hidden: int = 128
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 256
    seed: int = 0


def train_decoder(codes, actions, num_actions, codebook_size,
                  label_fraction: float, cfg: DecoderConfig | None = None,
                  val_codes=None, val_actions=None):
    """Train z->a decoder on a random ``label_fraction`` of (codes, actions).

    Returns ``(state, model, train_acc, val_acc)``. If a held-out set is not
    provided, accuracy is measured on the *full* dataset (codes are discrete so
    this still reflects how well the codebook separates actions).
    """
    cfg = cfg or DecoderConfig()
    codes = np.asarray(codes)
    actions = np.asarray(actions).astype(np.int32)
    n = len(codes)
    rng = np.random.default_rng(cfg.seed)

    n_label = max(num_actions, int(label_fraction * n))
    sel = rng.choice(n, size=min(n_label, n), replace=False)
    lab_codes, lab_actions = codes[sel], actions[sel]

    model = ActionDecoder(num_actions, codebook_size, cfg.hidden)
    key = jax.random.PRNGKey(cfg.seed)
    dummy = jnp.zeros((1, codebook_size))
    params = model.init(key, dummy)
    tx = optax.adam(cfg.lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    eye = np.eye(codebook_size, dtype=np.float32)

    @jax.jit
    def step(state, z, a):
        def loss_fn(p):
            logits = model.apply(p, z)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, a))
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    for _ in range(cfg.epochs):
        perm = rng.permutation(len(lab_codes))
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            z = jnp.asarray(eye[lab_codes[b]])
            a = jnp.asarray(lab_actions[b])
            state, _ = step(state, z, a)

    @jax.jit
    def predict(p, z):
        return jnp.argmax(model.apply(p, z), axis=-1)

    def accuracy(c, a):
        pred = np.asarray(predict(state.params, jnp.asarray(eye[c])))
        return float((pred == a).mean())

    train_acc = accuracy(lab_codes, lab_actions)
    if val_codes is not None:
        val_acc = accuracy(np.asarray(val_codes), np.asarray(val_actions).astype(np.int32))
    else:
        val_acc = accuracy(codes, actions)

    print(f"[decoder] label_frac={label_fraction:.2%} "
          f"(n={len(lab_codes):,})  train_acc={train_acc:.3f}  acc={val_acc:.3f}")
    return state, model, train_acc, val_acc


def label_budget_sweep(codes, actions, num_actions, codebook_size,
                       fractions=(0.01, 0.05, 0.25, 1.0),
                       cfg: DecoderConfig | None = None, seed: int = 0):
    """Run the full label-budget sweep and return a results table."""
    n = len(codes)
    split = int(0.9 * n)
    perm = np.random.default_rng(seed).permutation(n)
    tr, va = perm[:split], perm[split:]
    codes_tr, actions_tr = np.asarray(codes)[tr], np.asarray(actions)[tr]
    codes_va, actions_va = np.asarray(codes)[va], np.asarray(actions)[va]

    rows = []
    for f in fractions:
        _, _, train_acc, val_acc = train_decoder(
            codes_tr, actions_tr, num_actions, codebook_size, f, cfg,
            val_codes=codes_va, val_actions=actions_va)
        rows.append({"label_fraction": f, "train_acc": train_acc, "val_acc": val_acc})

    # majority-class baseline for context
    maj = np.bincount(np.asarray(actions_va), minlength=num_actions).max() / len(actions_va)
    print(f"[decoder] majority-class val accuracy = {maj:.3f}")
    return rows, maj
