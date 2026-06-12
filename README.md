# Latent Action World Models for Craftax

A world-model RL pipeline for **Craftax-classic** that learns a **discrete latent
action space** from unlabeled gameplay, pretrains a **Dreamer-style RSSM** on those
latent actions, decodes them to real actions with a small labeled subset, and
fine-tunes an actor-critic in imagination — then measures sample/label efficiency
vs. baselines. Implements `PLAN.md`.

## Quick start (Google Colab)

1. Upload this folder to Google Drive (or push to a GitHub repo).
2. Open `notebooks/craftax_latent_action.ipynb` in Colab.
3. **Runtime → Change runtime type → GPU** (T4 or better).
4. In the setup cell, set `CRAFTAX_PROJECT_ROOT` to where `src/` lives
   (e.g. your cloned repo dir, or a Drive path).
5. Run cells top to bottom. Each phase checkpoints to Drive so you can resume.

For a fast smoke test, shrink the `steps` / `total_timesteps` / `total_frames`
fields in the config cells (everything is a `NamedTuple` you can override inline).

## Layout

```
src/
  env_utils.py           Craftax env wrappers + jit/vmap vectorization, benchmark, score
  baseline_ppo.py        PureJaxRL-style PPO baseline (symbolic obs)         [Phase 1]
  collect.py             trajectory dataset collection + (o_t,o_t+1,a_t) pairs[Phase 1]
  latent_action_model.py CNN encoder + VQ bottleneck + forward decoder        [Phase 2]
  rssm.py                Dreamer-style RSSM + open-loop rollout error         [Phase 3]
  action_decoder.py      z->a MLP + label-budget sweep                        [Phase 4]
  agent.py               actor-critic trained in imagination + real-env eval  [Phase 4]
  eval.py                NMI / confusion / rollout / budget / score plots      [Phase 5]
  visualize.py           render gameplay: GIF/video/grid, LAM preds, imagination
  checkpoint.py          Drive mount + msgpack/pickle checkpointing
configs/default.yaml     all hyperparameters per phase
notebooks/craftax_latent_action.ipynb   the runnable Colab notebook
build_notebook.py        regenerates the notebook from source-of-truth cells
```

## Phases (maps to `PLAN.md`)

| Phase | What | Headline result |
|---|---|---|
| 1 | Env + PPO baseline + data | steps/sec benchmark, PPO reward curve, saved dataset |
| 2 | Latent action VQ-VAE | NMI(z;a), cluster purity, confusion matrix |
| 3 | RSSM world model | rollout MSE: latent vs. true actions |
| 4 | Decoder + imagination agent | label-budget accuracy curve, real-env score |
| 5 | Eval + ablations | score comparison, codebook-size ablation table |

## Design notes

- **Pure JAX everywhere** so Craftax's thousands of parallel envs actually pay off
  (`jax.vmap` + `jax.jit` + `lax.scan`). PPO and the RSSM training loops are single
  XLA programs.
- **Pixels for the latent-action model, symbolic for dynamics/RL.** The novel piece
  (Phase 2) needs frames; the RSSM and PPO use the fast flat observation, per the
  plan's "fall back to symbolic if pixels are too slow" guidance.
- **Latent codes never see action labels** until the Phase-4 decoder, which is the
  whole point — labels are spent only on the small `z→a` map.

## Regenerating the notebook

Edit `build_notebook.py` (the cell source-of-truth), then:

```bash
python build_notebook.py
```

## References

See `PLAN.md` → Key References (Craftax, LAPA, DreamerV3, …).
