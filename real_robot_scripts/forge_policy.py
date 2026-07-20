"""
Load a locally-trained FORGE SAC policy for real-robot evaluation.

Our checkpoints are saved per-agent to disk (no wandb artifacts) by
``learning/sac.py::write_checkpoint``:

    <run_dir>/<agent_idx>/checkpoints/ckpt_<step>.pt
    <run_dir>/<agent_idx>/config.yaml            (dumped ConfigManager config)

Each ``.pt`` dict holds a single-agent-sliced ``policy`` state, the per-agent
``observation_preprocessor`` state, and metadata. This module rebuilds a
``num_agents=1`` ``BlockSimBaActor`` from the dumped config, loads the slice, and
exposes a deterministic (tanh-of-mean) action call — the same continuous head
math as ``BlockSimBaActor.act`` but without sampling.
"""

import dataclasses
import glob
import os
import re

import torch

from models.block_simba import BlockSimBaActor, assign_block_slice
from real_robot_scripts.observation_builder import ObservationNormalizer


def _find_checkpoint(run_dir: str, agent_idx: int, step) -> str:
    """Resolve the ckpt_<step>.pt path (latest step if ``step`` is None)."""
    ckpt_dir = os.path.join(run_dir, str(agent_idx), "checkpoints")
    if not os.path.isdir(ckpt_dir):
        # Allow pointing --checkpoint directly at a single-agent folder.
        alt = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(alt):
            ckpt_dir = alt
        else:
            raise FileNotFoundError(f"No checkpoints/ dir under {run_dir}/{agent_idx}")

    if step is not None:
        path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    if not files:
        raise FileNotFoundError(f"No ckpt_*.pt in {ckpt_dir}")

    def step_of(p):
        m = re.search(r"ckpt_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(files, key=step_of)


class ForgePolicy:
    """A single trained FORGE actor + its frozen observation normalizer.

    Args:
        run_dir: Path to a training run directory (contains ``<agent>/checkpoints``)
                 or directly a single-agent folder.
        config: The resolved training config dict (from
                ``run_config.resolve_runtime_config``): ``model_cfg`` /
                ``sac_cfg`` / ``breakable_peg_cfg`` / ``force_obs_cfg``.
        agent_idx: Which block-parallel agent slot to load.
        step: Checkpoint step (None = latest).
        action_dim: Env action dimension (FORGE = 7).
        device: Torch device.
    """

    def __init__(self, run_dir: str, config: dict, agent_idx: int = 0, step=None,
                 action_dim: int = 7, device: str = "cpu"):
        self.device = device
        self.action_dim = int(action_dim)

        ckpt_path = _find_checkpoint(run_dir, agent_idx, step)
        print(f"[ForgePolicy] checkpoint: {ckpt_path}")

        self.model_cfg = config["model_cfg"]
        self.sac_cfg = config["sac_cfg"]
        # Training-derived break threshold (authoritative; e.g. 10 N).
        self.break_force = float(config["breakable_peg_cfg"].break_force)
        force_obs = config["force_obs_cfg"]
        if getattr(force_obs, "history_enabled", False):
            raise NotImplementedError(
                "This checkpoint trained with force-obs history "
                f"(history_length={force_obs.history_length}); the real ObservationBuilder "
                "assembles only the stock FORGE obs. Add history support before evaluating it."
            )

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        for key in ("policy", "observation_preprocessor"):
            if key not in ckpt:
                raise RuntimeError(f"Checkpoint {ckpt_path} missing '{key}'")
        self.step = int(ckpt.get("step", step if step is not None else -1))

        # obs_dim is authoritative from the frozen normalizer stats.
        self.normalizer = ObservationNormalizer(
            ckpt["observation_preprocessor"], device=device,
        )
        self.obs_dim = self.normalizer.obs_dim

        actor_kwargs = dataclasses.asdict(self.model_cfg.actor)
        self.policy = BlockSimBaActor(
            observation_space=self.obs_dim,
            action_space=self.action_dim,
            device=device,
            num_agents=1,
            predict_success=self.sac_cfg.predict_success,
            **actor_kwargs,
        ).to(device)

        # ckpt["policy"] is already a single-agent slice; write it into slot 0.
        assign_block_slice(self.policy, 0, 1, ckpt["policy"])
        self.policy.eval()

        print(f"[ForgePolicy] loaded step={self.step}, obs_dim={self.obs_dim}, "
              f"action_dim={self.action_dim}, "
              f"continuous_dims={self.policy.continuous_dims}, "
              f"force_zero_dims={self.policy.force_zero_dims}, "
              f"predict_success={self.sac_cfg.predict_success}, "
              f"break_force={self.break_force} N")

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, std_scale: float = 0.0) -> torch.Tensor:
        """Return the [action_dim] env-facing action for a single raw observation.

        Deterministic by default (``std_scale=0``): the continuous head is
        ``tanh(mean)`` scattered into ``continuous_dims`` with force-zero dims left
        at 0 — identical to ``BlockSimBaActor.act`` minus the Gaussian sample.
        ``std_scale > 0`` adds ``std_scale * exp(log_std)`` pre-tanh noise for a
        stochastic rollout.
        """
        norm_obs = self.normalizer.normalize(obs.unsqueeze(0))  # [1, obs_dim]
        raw_out, outputs = self.policy.compute({"observations": norm_obs}, role="policy")

        cont_mean = raw_out.index_select(-1, self.policy._cont_out_idx)  # [1, num_continuous]
        if std_scale > 0.0:
            log_std = outputs["log_std"]
            log_std = torch.clamp(log_std, self.policy._g_min_log_std, self.policy._g_max_log_std)
            u = cont_mean + std_scale * log_std.exp() * torch.randn_like(cont_mean)
        else:
            u = cont_mean
        a_cont = torch.tanh(u)

        actions = raw_out.new_zeros((1, self.policy.num_actions))
        actions.index_copy_(-1, self.policy._cont_action_idx, a_cont)
        return actions[0]
