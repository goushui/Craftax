"""Evaluation, metrics, and plotting (Phases 2-5).

Self-contained metrics so the notebook can produce every headline figure:
  * latent-code <-> true-action confusion matrix + normalized mutual information
    + cluster-purity (Phase 2 validation, the first concrete result);
  * RSSM open-loop rollout-error curves, latent vs. true action (Phase 3);
  * label-budget accuracy curve (Phase 4);
  * sample-efficiency / score comparison bars (Phase 5).

Only depends on numpy + matplotlib so it runs even if JAX training is skipped.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Phase 2: latent code vs. true action
# ---------------------------------------------------------------------------

def confusion_matrix(codes, actions, codebook_size, num_actions):
    """(codebook_size, num_actions) counts of (z_t, a_t) co-occurrence."""
    codes = np.asarray(codes).astype(int)
    actions = np.asarray(actions).astype(int)
    cm = np.zeros((codebook_size, num_actions), dtype=np.int64)
    np.add.at(cm, (codes, actions), 1)
    return cm


def normalized_mutual_info(codes, actions):
    """NMI(z; a) in [0,1]. 0 = independent, 1 = z perfectly determines a."""
    codes = np.asarray(codes).astype(int)
    actions = np.asarray(actions).astype(int)
    n = len(codes)
    kz, ka = codes.max() + 1, actions.max() + 1
    joint = np.zeros((kz, ka))
    np.add.at(joint, (codes, actions), 1)
    joint /= n
    pz = joint.sum(1, keepdims=True)
    pa = joint.sum(0, keepdims=True)

    nz = joint > 0
    mi = np.sum(joint[nz] * np.log(joint[nz] / (pz @ pa)[nz]))

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    hz, ha = entropy(pz.ravel()), entropy(pa.ravel())
    denom = 0.5 * (hz + ha)
    return float(mi / denom) if denom > 0 else 0.0


def cluster_purity(codes, actions):
    """Fraction of frames whose code's majority action matches their action."""
    codes = np.asarray(codes).astype(int)
    actions = np.asarray(actions).astype(int)
    total, correct = len(codes), 0
    for c in np.unique(codes):
        mask = codes == c
        if mask.sum() == 0:
            continue
        maj = np.bincount(actions[mask]).max()
        correct += maj
    return correct / total


def plot_confusion(cm, title="Latent code vs. true action", normalize=True, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    m = cm.astype(float)
    if normalize:
        m = m / (m.sum(1, keepdims=True) + 1e-9)
    im = ax.imshow(m, aspect="auto", cmap="viridis")
    ax.set_xlabel("true action a_t")
    ax.set_ylabel("latent code z_t")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax


def report_latent_quality(codes, actions, codebook_size, num_actions, plot=True):
    nmi = normalized_mutual_info(codes, actions)
    purity = cluster_purity(codes, actions)
    used = len(np.unique(codes))
    print(f"[eval] NMI(z;a)={nmi:.3f}  cluster_purity={purity:.3f}  "
          f"codes_used={used}/{codebook_size}")
    if plot:
        cm = confusion_matrix(codes, actions, codebook_size, num_actions)
        plot_confusion(cm)
    return {"nmi": nmi, "purity": purity, "codes_used": used}


# ---------------------------------------------------------------------------
# Phase 3: rollout error comparison
# ---------------------------------------------------------------------------

def plot_rollout_comparison(mse_latent, mse_true, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    steps = np.arange(1, len(mse_latent) + 1)
    ax.plot(steps, mse_latent, "-o", label="RSSM (latent actions)")
    ax.plot(steps, mse_true, "-s", label="RSSM (true actions)")
    ax.set_xlabel("imagination step")
    ax.set_ylabel("open-loop obs MSE (symlog space)")
    ax.set_title("World-model rollout error: latent vs. true actions")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Phase 4: label-budget curve
# ---------------------------------------------------------------------------

def plot_label_budget(rows, majority=None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    fr = [r["label_fraction"] for r in rows]
    acc = [r["val_acc"] for r in rows]
    ax.plot(np.array(fr) * 100, acc, "-o", label="z->a decoder val acc")
    if majority is not None:
        ax.axhline(majority, ls="--", color="gray", label="majority class")
    ax.set_xscale("log")
    ax.set_xlabel("label budget (% of frames)")
    ax.set_ylabel("decoding accuracy")
    ax.set_title("Action-decoder label efficiency")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Phase 5: final score comparison
# ---------------------------------------------------------------------------

def plot_score_comparison(scores: dict, ax=None):
    """scores: {name -> craftax_score}. Bar chart of the headline comparison."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    ax.bar(names, vals, color=["#888", "#4c78a8", "#54a24b"][:len(names)])
    ax.set_ylabel("Craftax-classic score")
    ax.set_title("Final comparison")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return ax


def ppo_score_curve(metrics, num_envs, num_steps):
    """Turn PPO scan metrics into an (env_steps, mean_reward) curve."""
    rewards = np.asarray(metrics["reward"])
    env_steps = np.arange(1, len(rewards) + 1) * num_envs * num_steps
    return env_steps, rewards


def plot_sample_efficiency(curves: dict, ax=None):
    """curves: {name -> (env_steps, metric)}. Sample-efficiency comparison."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for name, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=name)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("mean reward / score")
    ax.set_title("Sample efficiency")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def ablation_table(results: list[dict]) -> str:
    """Render a list of {key:val} ablation rows as a markdown table string."""
    if not results:
        return ""
    cols = list(results[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for r in results:
        lines.append("| " + " | ".join(
            f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    table = "\n".join(lines)
    print(table)
    return table
