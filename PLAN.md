# Project: Latent Action World Models for Craftax

## Overview

Build a world-model-based RL pipeline for **Craftax-classic** (a fast, JAX-based, Minecraft-inspired open-world survival benchmark) that:

1. Learns a **discrete latent action space** from unlabeled gameplay trajectories (LAPA/Genie-style VQ-VAE on frame pairs) — no ground-truth action labels used at this stage.
2. Pretrains a **Dreamer-style RSSM world model** on unlabeled trajectories using the learned latent actions instead of true actions.
3. Trains a small **decoder mapping latent actions → real Craftax actions** using only a small labeled subset (sweep over label budget: 1%, 5%, 25%, 100%).
4. Fine-tunes an **actor-critic agent in imagination**, initialized from the pretrained world model.
5. Evaluates sample efficiency vs. baselines (PPO from scratch, DreamerV3 from scratch) on the Craftax-classic score (% of 22 achievements).

This is a 4–6 week project. Target environment: Google Colab (free/Pro tier, single T4/A100 GPU). Full background and motivation are in `Craftax_Latent_Action_Project_Plan.md` in this same folder — read that first for context on *why*, this file is about *how* to build it.

---

## Environment Setup

- Python 3.10+, JAX with CUDA support matching the runtime's CUDA version (check before installing — version mismatches are the #1 source of breakage on Colab).
- `pip install craftax` (or clone from the Craftax GitHub repo if the PyPI package lags).
- Use `jax.vmap` / `jax.jit` to vectorize across thousands of parallel environments — this is the whole point of Craftax's speed.
- Checkpoint model + optimizer state every N steps (e.g., every 50k env steps) to survive Colab disconnects. Save to a mounted Drive folder, not local Colab storage.

---

## Repo Structure (target)

```
craftax-latent-action/
├── CLAUDE.md                  # this file
├── README.md                  # results, plots, demo gifs (write last)
├── configs/                    # hyperparameter configs per phase
├── data/
│   └── trajectories/           # collected rollout datasets (frames, actions, rewards)
├── src/
│   ├── env_utils.py             # Craftax env wrappers, vectorization helpers
│   ├── baseline_ppo.py          # baseline PPO agent (Phase 1)
│   ├── latent_action_model.py   # CNN encoder + VQ bottleneck + forward decoder (Phase 2)
│   ├── rssm.py                  # Dreamer-style recurrent state-space model (Phase 3)
│   ├── action_decoder.py        # latent -> real action MLP (Phase 4)
│   ├── agent.py                 # actor-critic trained in imagination (Phase 4)
│   └── eval.py                  # sample-efficiency curves, ablations (Phase 5)
├── notebooks/                   # Colab-friendly notebooks per phase
└── checkpoints/                 # model checkpoints (gitignored, save to Drive)
```

---

## Implementation Phases

### Phase 1 — Baseline + data collection
- [ ] Set up Craftax-classic env, confirm GPU vectorization works (benchmark steps/sec).
- [ ] Implement/port a baseline PPO agent; verify it reproduces published numbers (~human-level on a subset of the 22 achievements).
- [ ] Collect a trajectory dataset: frames + true actions + rewards from (a) random policy, (b) partially-trained PPO. Target 1–5M frames.
- **Done when:** baseline PPO trains and scores reasonably; trajectory dataset saved to `data/trajectories/`.

### Phase 2 — Latent action model
- [ ] Implement small CNN encoder over frame pairs `(o_t, o_{t+1})` → VQ bottleneck → discrete latent code `z_t`. Codebook size K ≈ 16–32 (Craftax-classic has 17 actions).
- [ ] Train with forward-prediction/reconstruction loss (predict `o_{t+1}` from `o_t, z_t`) + VQ commitment loss.
- [ ] Validate: compute confusion matrix / normalized mutual information between `z_t` and true `a_t`. This is the first concrete result — report it even if imperfect.
- **Done when:** latent codes show measurable correlation with true actions; metrics + visualizations saved.

### Phase 3 — World model pretraining on latent actions
- [ ] Implement a simplified RSSM (Dreamer-style).
- [ ] Pretrain RSSM on unlabeled trajectories using `z_t` as the action input.
- [ ] Compare multi-step rollout/reconstruction error vs. an RSSM trained with true actions (sanity check on information loss).
- **Done when:** RSSM trained on latent actions achieves reasonable rollout quality relative to the true-action RSSM.

### Phase 4 — Action decoder + agent fine-tuning
- [ ] Train `z_t → a_t` MLP decoder on labeled subsets at multiple budgets (1%, 5%, 25%, 100%). Plot accuracy vs. label budget.
- [ ] Initialize actor-critic from the pretrained RSSM; fine-tune end-to-end with real environment interaction (Dreamer-style imagination training).
- **Done when:** agent trains successfully; sample-efficiency curve (score vs. env steps) recorded.

### Phase 5 — Evaluation & ablations
- [ ] Compare: PPO from scratch vs. DreamerV3 from scratch vs. latent-action-pretrained agent (all on Craftax-classic score + sample efficiency).
- [ ] Ablations: codebook size, unlabeled pretraining data volume, label budget, with/without latent-action pretraining.
- [ ] (Stretch) zero-shot test of pretrained world model on full Craftax.
- **Done when:** comparison plots + ablation table complete.

### Phase 6 — Writeup
- [ ] README with sample-efficiency plots, latent action t-SNE/cluster viz, confusion matrices, rollout GIFs.
- [ ] Short (4–6 page) report: motivation, method, results, discussion, limitations, future work.
- [ ] One-paragraph elevator pitch + key figure for lab applications.

---

## Fallback / Risk Notes

- If latent actions don't correlate well with true actions: still report the label-efficiency curve as the core finding (representation-learning contribution, not pure RL score).
- If end-to-end RL fine-tuning underperforms baselines: lead with the pretraining/analysis results (Phases 2–3) rather than final RL score.
- If pixel-based pipeline is too slow: fall back to symbolic Craftax observations for the RL agent; keep pixel-based latent action learning as the core novel piece.
- Skip full Craftax (only attempt Craftax-classic) if time runs short — it's a complete, sufficient scope on its own.

---

## Key References

- Craftax: https://arxiv.org/pdf/2402.16801
- LAPA (Latent Action Pretraining from Videos): https://arxiv.org/abs/2410.11758, https://github.com/LatentActionPretraining/LAPA
- ViPRA: https://arxiv.org/abs/2511.07732
- Latent Particle World Models (LPWM): https://arxiv.org/abs/2603.04553
- ITC (current Craftax SOTA): https://arxiv.org/html/2605.16457
- DreamerV3 — baseline RSSM architecture (search for paper/repo)
