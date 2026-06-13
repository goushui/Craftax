"""Generate notebooks/craftax_latent_action.ipynb from a list of cells.

Keeping the notebook as a Python source-of-truth avoids hand-editing JSON and
makes regenerating after code changes trivial. Run:  python build_notebook.py
"""

import json
import os

MD = "markdown"
CODE = "code"


def cell(kind, source):
    src = source if isinstance(source, list) else source.splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == CODE:
        base["outputs"] = []
        base["execution_count"] = None
    return base


cells = []
def md(s): cells.append(cell(MD, s.strip("\n")))
def code(s): cells.append(cell(CODE, s.strip("\n")))


# ===========================================================================
md(r"""
# Latent Action World Models for Craftax

End-to-end Colab pipeline implementing `PLAN.md`:

1. **Phase 1** — Craftax-classic setup, PPO baseline, trajectory data collection
2. **Phase 2** — Latent action VQ-VAE on frame pairs (the novel piece)
3. **Phase 3** — Dreamer-style RSSM world model on latent actions
4. **Phase 4** — z→a action decoder (label-budget sweep) + imagination agent
5. **Phase 5** — Evaluation, comparison, ablations

**Runtime:** set **Runtime → Change runtime type → GPU (T4 or better)** before running.
Each phase saves checkpoints to Google Drive so you can resume after disconnects.
""")

# ---------------------------------------------------------------------------
md("## 0 · Setup\n\nInstall dependencies and clone/place the `src/` package. Run this cell first.")

code(r"""
# JAX with CUDA matching the Colab runtime (CUDA 12 as of 2025/2026 Colab images).
# If you hit a CUDA mismatch, check `!nvcc --version` and pick the matching jax build.
!pip -q install -U "jax[cuda12]"
!pip -q install -U craftax flax optax distrax matplotlib imageio
import jax
print("JAX devices:", jax.devices())
""")

code(r"""
# Clone the project repo so the src/ package is importable.
import sys, os
REPO_URL = "https://github.com/goushui/Craftax.git"
PROJECT_ROOT = "/content/Craftax"
if not os.path.isdir(PROJECT_ROOT):
    !git clone $REPO_URL $PROJECT_ROOT
else:
    !cd $PROJECT_ROOT && git pull --ff-only
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
%cd $PROJECT_ROOT

import importlib
import src.env_utils, src.baseline_ppo, src.collect, src.latent_action_model
import src.rssm, src.action_decoder, src.agent, src.eval, src.checkpoint, src.visualize
for m in [src.env_utils, src.baseline_ppo, src.collect, src.latent_action_model,
          src.rssm, src.action_decoder, src.agent, src.eval, src.checkpoint,
          src.visualize]:
    importlib.reload(m)
print("src package loaded.")
""")

code(r"""
# Storage for checkpoints/data/results. Default: Google DRIVE, so everything
# persists across runtime disconnects/timeouts. The first run pops an auth
# dialog; after that it's automatic. Set USE_DRIVE = False to use a local
# runtime dir instead (no popup, but wiped on disconnect).
USE_DRIVE = True
if USE_DRIVE:
    from src.checkpoint import mount_drive
    BASE = mount_drive("craftax-latent-action")
else:
    BASE = os.path.abspath("artifacts")
    for sub in ("checkpoints", "data/trajectories", "results"):
        os.makedirs(os.path.join(BASE, sub), exist_ok=True)
    print("artifacts base =", BASE)
CKPT = os.path.join(BASE, "checkpoints")
DATA = os.path.join(BASE, "data/trajectories")
RESULTS = os.path.join(BASE, "results")
print("checkpoints ->", CKPT)
print("datasets    ->", DATA)
print("results     ->", RESULTS)


def savefig(name):
    # Save the current matplotlib figure into RESULTS/ as a PNG, then show it.
    import matplotlib.pyplot as plt
    path = os.path.join(RESULTS, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print("[fig] saved ->", path)
    plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 0b · Watch the game play out

Before any training, render a rollout so you can actually *see* Craftax. A random
policy already wanders, mines, and dies — good for confirming the renderer works.
Re-run later with a trained policy to watch the agent improve.
""")

code(r"""
# Quick inline video of a random rollout (falls back to a frame grid if the
# video codec isn't available). Also saves a GIF to Drive.
from src.visualize import quick_watch
quick_watch(num_steps=200, fps=10, seed=0, as_video=True,
            save_path=os.path.join(RESULTS, "random_rollout.gif"))
""")

code(r"""
# Static frame grid (every Nth frame) — handy for figures / quick scrubbing.
from src.visualize import rollout_frames, show_grid
frames, actions, rewards = rollout_frames(num_steps=120, seed=3)
print("total reward:", rewards.sum())
show_grid(frames[::4], cols=8, titles=actions[::4]); import matplotlib.pyplot as plt; savefig("rollout_grid.png")
""")

# ---------------------------------------------------------------------------
md(r"""
## Phase 1 · Baseline + data collection

* Confirm GPU vectorization works (steps/sec benchmark).
* Train a PPO baseline on the symbolic observation.
* Collect a pixel-observation trajectory dataset (random + PPO policy) for Phases 2–4.

**Done when:** PPO trains and scores reasonably; dataset saved to Drive.
""")

code(r"""
# 1a. Vectorization sanity check — should be hundreds of thousands to millions of steps/sec.
from src.env_utils import benchmark_steps_per_sec
_ = benchmark_steps_per_sec(num_envs=1024, num_steps=200, obs_mode="symbolic")
_ = benchmark_steps_per_sec(num_envs=256,  num_steps=100, obs_mode="pixels")
""")

code(r"""
# 1b. Train PPO baseline. Reduce total_timesteps for a quick smoke test (e.g. 200_000).
from src.baseline_ppo import PPOConfig, train_ppo
ppo_cfg = PPOConfig(total_timesteps=2_000_000, num_envs=1024, num_steps=64, seed=0)
ppo_state, ppo_metrics, ppo_bundle = train_ppo(ppo_cfg)

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(7,4))
plt.plot(np.asarray(ppo_metrics["reward"]))
plt.xlabel("update"); plt.ylabel("mean step reward"); plt.title("PPO baseline training")
plt.grid(alpha=0.3); savefig("ppo_training_curve.png")

from src.checkpoint import save_state
save_state(ppo_state, os.path.join(CKPT, "ppo_baseline.msgpack"))
""")

code(r"""
# 1c. Collect trajectory datasets (pixels) under random + PPO(+epsilon) policies.
from src.collect import collect_dataset, _random_policy, _ppo_policy, make_pair_dataset
from src.env_utils import make_craftax, get_dims

env_p, params_p = make_craftax("pixels", auto_reset=True)
_, num_actions = get_dims(env_p, params_p)

# Random-policy data (broad coverage)
rand_data = collect_dataset(
    _random_policy(num_actions), total_frames=300_000, num_envs=256,
    obs_mode="pixels", seed=1,
    save_path=os.path.join(DATA, "random_pixels.npz"))

# PPO-policy data (on-distribution states). The PPO net was trained on symbolic
# obs, so we sample actions from a fresh epsilon-greedy-ish *random* policy here
# to keep it simple; swap in a pixel-trained PPO for higher-quality coverage.
ppo_like = collect_dataset(
    _random_policy(num_actions), total_frames=200_000, num_envs=256,
    obs_mode="pixels", seed=2,
    save_path=os.path.join(DATA, "ppo_pixels.npz"))

# Merge + build (o_t, o_t+1, a_t) pairs for the latent-action model.
import numpy as np
merged = {k: np.concatenate([rand_data[k], ppo_like[k]])
          for k in ["frames", "actions", "rewards", "dones"]}
merged["num_actions"] = rand_data["num_actions"]
pairs = make_pair_dataset(merged)
OBS_SHAPE = merged["frames"].shape[1:]   # (H, W, 3)
print("obs shape:", OBS_SHAPE, "num_actions:", int(merged["num_actions"]))
""")

# ---------------------------------------------------------------------------
md(r"""
## Phase 2 · Latent action model (VQ-VAE on frame pairs)

Encode `(o_t, o_{t+1})` → discrete code `z_t`, decode forward to predict `o_{t+1}`.
Then validate `z_t` against the held-out true actions via **NMI**, **cluster purity**,
and a **confusion matrix**. This is the first concrete result.

**Done when:** latent codes show measurable correlation with true actions.
""")

code(r"""
from src.latent_action_model import LAMConfig, train_lam, encode_codes
lam_cfg = LAMConfig(codebook_size=32, code_dim=64, steps=15_000, batch_size=256)
lam_model, lam_state, lam_hist = train_lam(pairs, OBS_SHAPE, lam_cfg)

from src.checkpoint import save_state
save_state(lam_state, os.path.join(CKPT, "lam.msgpack"))
""")

code(r"""
# Encode every pair to its discrete code, then score against true actions.
codes = encode_codes(lam_model, lam_state, pairs["o_t"], pairs["o_tp1"])
true_actions = pairs["a_t"]

from src.eval import report_latent_quality
NUM_ACTIONS = int(merged["num_actions"])
q = report_latent_quality(codes, true_actions, lam_cfg.codebook_size, NUM_ACTIONS, plot=True)
import matplotlib.pyplot as plt; savefig("phase2_confusion_matrix.png")
""")

code(r"""
# Visualize what the latent action model *predicts*: o_t, true o_t+1, and the
# decoder's predicted o_t+1 (with its discrete code z). Blurry-but-directional
# predictions are normal; look for the predicted frame moving in the right way.
from src.visualize import show_lam_predictions
show_lam_predictions(lam_model, lam_state, pairs, n=6); savefig("phase2_lam_predictions.png")
""")

# ---------------------------------------------------------------------------
md(r"""
## Phase 3 · World-model pretraining on latent actions

Train a Dreamer-style RSSM on the **symbolic** trajectory using `z_t` as the
action input, and a second RSSM using the **true** actions. Compare open-loop
rollout error to measure how much information the latent quantization loses.

> We need a symbolic-observation stream aligned with the latent codes. We
> re-collect a symbolic dataset with the *same seeds/policy* so action indices
> line up, then attach the latent codes computed from the pixel pairs.
""")

code(r"""
# Collect aligned symbolic data (same policy/seeds as the pixel collection so the
# action stream matches). For the latent-code stream we reuse `codes` from Phase 2.
from src.collect import collect_dataset, _random_policy
sym_rand = collect_dataset(_random_policy(num_actions), total_frames=300_000,
                           num_envs=256, obs_mode="symbolic", seed=1)
sym_ppo  = collect_dataset(_random_policy(num_actions), total_frames=200_000,
                           num_envs=256, obs_mode="symbolic", seed=2)
import numpy as np
sym = {k: np.concatenate([sym_rand[k], sym_ppo[k]]) for k in
       ["frames", "actions", "rewards", "dones"]}
OBS_DIM = sym["frames"].shape[1]
print("symbolic obs_dim:", OBS_DIM)

# Build the latent-code action stream aligned to the symbolic transitions.
# codes were computed on non-boundary pairs; map them back to a per-step stream.
valid = ~merged["dones"][:-1]
code_stream = np.zeros(len(merged["actions"]) - 1, dtype=np.int32)
code_stream[valid] = codes
# pad to symbolic length (they share length up to the trailing transition)
code_stream = np.concatenate([code_stream, code_stream[-1:]])
code_stream = code_stream[:len(sym["actions"])]
""")

code(r"""
from src.rssm import RSSMConfig, train_rssm, rollout_error

# RSSM trained on LATENT actions
rssm_cfg_latent = RSSMConfig(obs_dim=OBS_DIM, num_inputs=lam_cfg.codebook_size,
                             seq_len=32, batch_size=32, steps=12_000)
rssm_model_L, rssm_state_L, hist_L = train_rssm(
    sym["frames"], code_stream, sym["rewards"], sym["dones"], rssm_cfg_latent)

# RSSM trained on TRUE actions (baseline)
rssm_cfg_true = rssm_cfg_latent._replace(num_inputs=NUM_ACTIONS)
rssm_model_T, rssm_state_T, hist_T = train_rssm(
    sym["frames"], sym["actions"], sym["rewards"], sym["dones"], rssm_cfg_true)

from src.checkpoint import save_state
save_state(rssm_state_L, os.path.join(CKPT, "rssm_latent.msgpack"))
save_state(rssm_state_T, os.path.join(CKPT, "rssm_true.msgpack"))
""")

code(r"""
# Open-loop rollout-error comparison.
mse_L = rollout_error(rssm_model_L, rssm_state_L, sym["frames"], code_stream,
                      sym["rewards"], sym["dones"], rssm_cfg_latent, horizon=15)
mse_T = rollout_error(rssm_model_T, rssm_state_T, sym["frames"], sym["actions"],
                      sym["rewards"], sym["dones"], rssm_cfg_true, horizon=15)

from src.eval import plot_rollout_comparison
import matplotlib.pyplot as plt
plot_rollout_comparison(mse_L, mse_T); savefig("phase3_rollout_comparison.png")
""")

code(r"""
# 'Watch' the symbolic world model: does its imagined reward track reality over
# an open-loop rollout? (The symbolic RSSM has no image to render, so reward is
# the readable signal.)
from src.visualize import imagination_vs_real_rewards
imagination_vs_real_rewards(rssm_model_L, rssm_state_L, rssm_cfg_latent,
                            sym["frames"], code_stream, sym["rewards"],
                            sym["dones"], horizon=20); savefig("phase3_imagination_vs_real.png")
""")

# ---------------------------------------------------------------------------
md(r"""
## Phase 4 · Action decoder + imagination agent

* Train `z_t → a_t` MLP at label budgets {1%, 5%, 25%, 100%} → label-efficiency curve.
* Train an actor-critic **in the pretrained RSSM's imagination**, then evaluate in the real env.

**Done when:** decoder curve recorded and agent trains / produces a real-env score.
""")

code(r"""
from src.action_decoder import DecoderConfig, label_budget_sweep, train_decoder
dec_cfg = DecoderConfig(epochs=40)
rows, majority = label_budget_sweep(codes, true_actions, NUM_ACTIONS,
                                    lam_cfg.codebook_size,
                                    fractions=(0.01, 0.05, 0.25, 1.0), cfg=dec_cfg)

from src.eval import plot_label_budget
import matplotlib.pyplot as plt
plot_label_budget(rows, majority); savefig("phase4_label_budget.png")

# Fit a final decoder on 100% labels to get the code->action lookup for eval.
import numpy as np
dec_state, dec_model, _, _ = train_decoder(
    codes, true_actions, NUM_ACTIONS, lam_cfg.codebook_size, 1.0, dec_cfg)
eye = np.eye(lam_cfg.codebook_size, dtype=np.float32)
code_to_action = np.asarray(
    dec_model.apply(dec_state.params, eye).argmax(-1))
print("code -> action map:", code_to_action)
""")

code(r"""
# Train actor-critic in the latent-action RSSM's imagination, then evaluate.
from src.agent import AgentConfig, train_agent_in_imagination, evaluate_in_env
ag_cfg = AgentConfig(steps=3_000, imag_horizon=15)
ac_model, ac_state, ag_hist = train_agent_in_imagination(
    rssm_model_L, rssm_state_L, rssm_cfg_latent,
    sym["frames"], code_stream, sym["rewards"], sym["dones"], ag_cfg)

score_latent, rates = evaluate_in_env(
    ac_model, ac_state, rssm_model_L, rssm_state_L, rssm_cfg_latent,
    code_to_action, num_envs=64, num_steps=512)

from src.checkpoint import save_state
save_state(ac_state, os.path.join(CKPT, "agent_latent.msgpack"))
""")

# ---------------------------------------------------------------------------
md(r"""
## Phase 5 · Evaluation & ablations

* Compare PPO-from-scratch vs. the latent-action-pretrained agent.
* Ablate codebook size and pretraining-data volume (re-run Phases 2–4 inside a loop).
* Assemble the headline figures and a markdown ablation table.
""")

code(r"""
# Headline comparison (extend with DreamerV3-from-scratch if you implement it).
from src.eval import plot_score_comparison, plot_sample_efficiency, ppo_score_curve
import matplotlib.pyplot as plt

# Approximate PPO "score" proxy from its reward curve final value, plus the
# measured latent-agent score. Replace ppo proxy with a real achievement eval
# for the paper figure.
ppo_final = float(np.asarray(ppo_metrics["reward"])[-50:].mean())
scores = {
    "PPO (proxy reward)": ppo_final,
    "Latent-pretrained agent": score_latent,
}
plot_score_comparison(scores); savefig("phase5_score_comparison.png")

xs, ys = ppo_score_curve(ppo_metrics, ppo_cfg.num_envs, ppo_cfg.num_steps)
plot_sample_efficiency({"PPO": (xs, ys)}); savefig("phase5_sample_efficiency.png")
""")

code(r"""
# Ablation: codebook size K vs. latent-action quality (NMI / purity / decoder acc).
from src.latent_action_model import LAMConfig, train_lam, encode_codes
from src.eval import normalized_mutual_info, cluster_purity, ablation_table
from src.action_decoder import label_budget_sweep

abl_rows = []
for K in (8, 16, 32, 64):
    cfg_k = LAMConfig(codebook_size=K, code_dim=64, steps=6_000, batch_size=256)
    m_k, s_k, _ = train_lam(pairs, OBS_SHAPE, cfg_k, log_every=2000)
    codes_k = encode_codes(m_k, s_k, pairs["o_t"], pairs["o_tp1"])
    nmi = normalized_mutual_info(codes_k, true_actions)
    pur = cluster_purity(codes_k, true_actions)
    rows_k, _ = label_budget_sweep(codes_k, true_actions, NUM_ACTIONS, K,
                                   fractions=(1.0,))
    abl_rows.append({"codebook_K": K, "nmi": nmi, "purity": pur,
                     "decoder_acc_100pct": rows_k[0]["val_acc"]})

table_md = ablation_table(abl_rows)
with open(os.path.join(RESULTS, "ablation_codebook.md"), "w") as f:
    f.write(table_md)
""")

md(r"""
## Phase 6 · Writeup (manual)

Collect the saved figures from `RESULTS/` into the README:
* confusion matrix + NMI (Phase 2)
* rollout-error comparison (Phase 3)
* label-budget curve (Phase 4)
* score comparison + ablation table (Phase 5)

See `PLAN.md` Phase 6 for the report outline.
""")

# ===========================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "notebooks",
                   "craftax_latent_action.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"wrote {out} ({len(cells)} cells)")
