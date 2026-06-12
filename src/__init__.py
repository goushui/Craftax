"""Latent Action World Models for Craftax -- source package.

Phase modules:
  env_utils           -- Craftax env wrappers + vectorization (Phase 1)
  baseline_ppo        -- PPO baseline agent (Phase 1)
  collect             -- trajectory dataset collection (Phase 1)
  latent_action_model -- VQ-VAE over frame pairs (Phase 2)
  rssm                -- Dreamer-style world model (Phase 3)
  action_decoder      -- latent -> real action MLP (Phase 4)
  agent               -- actor-critic in imagination (Phase 4)
  eval                -- metrics + plots (Phases 2-5)
  checkpoint          -- Drive checkpointing
"""

__all__ = [
    "env_utils", "baseline_ppo", "collect", "latent_action_model",
    "rssm", "action_decoder", "agent", "eval", "checkpoint",
]
