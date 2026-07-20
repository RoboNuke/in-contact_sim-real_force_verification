"""
Observation builder for real-robot FORGE peg-insert evaluation.

Assembles the exact policy observation the FORGE env feeds the actor during
training (`isaaclab_tasks/direct/forge/forge_env.py::_get_observations`), from a
real-robot `StateSnapshot`, and applies the frozen training normalizer.

FORGE policy obs layout (`cfg.obs_order` + `prev_actions`), 24-D:

    fingertip_pos_rel_fixed (3)   ee_pos - goal (hole-entrance obs frame)
    fingertip_quat          (4)   measured EE quat (w,x,y,z)
    ee_linvel               (3)
    ee_angvel               (3)
    ft_force                (3)    measured contact force (EE/body frame, env-on-robot)
    force_threshold         (1)    constant contact-penalty threshold (eval)
    prev_actions            (7)    previous EMA-smoothed action, dims 3:5 zeroed

Pure PyTorch — no Isaac Sim dependency. Mirrors the reference repo's
`real_robot_exps/observation_builder.py`, retargeted to the FORGE obs_order.
"""

import torch


# Per-term dims — must match Isaac Lab factory/forge OBS_DIM_CFG.
OBS_DIM_MAP = {
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "ft_force": 3,
    "force_threshold": 1,
}

# The stock FORGE obs_order (forge_env_cfg.py). Kept as the default so the builder
# reproduces training without extra configuration.
FORGE_OBS_ORDER = [
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "ft_force",
    "force_threshold",
]


class ObservationBuilder:
    """Builds the 24-D FORGE policy observation from a real-robot state snapshot.

    On hardware, natural sensor noise replaces the sim's injected Gaussian noise,
    so no noise is added here — the measured values are used directly.

    Args:
        force_threshold: The fixed contact-penalty threshold scalar (N) the policy
            sees in the ``force_threshold`` obs channel. In sim this is sampled per
            episode from ``contact_penalty_threshold_range`` ([5, 10] default); at
            eval a single constant (its midpoint, 7.5) is used.
        action_dim: Env action dimension (7 for FORGE) — width of prev_actions.
        obs_order: Observation term order (defaults to the stock FORGE order).
        device: Torch device.
    """

    def __init__(
        self,
        force_threshold: float,
        action_dim: int = 7,
        obs_order: list = None,
        device: str = "cpu",
    ):
        self.obs_order = list(obs_order) if obs_order is not None else list(FORGE_OBS_ORDER)
        self.force_threshold = float(force_threshold)
        self.action_dim = int(action_dim)
        self.device = device

        for name in self.obs_order:
            if name not in OBS_DIM_MAP:
                raise ValueError(
                    f"Unknown observation term '{name}' in obs_order. "
                    f"Known terms: {list(OBS_DIM_MAP.keys())}"
                )

        self.obs_dim = sum(OBS_DIM_MAP[n] for n in self.obs_order) + self.action_dim

        print(f"[ObservationBuilder] obs_order={self.obs_order}")
        idx = 0
        for name in self.obs_order:
            dim = OBS_DIM_MAP[name]
            print(f"  [{idx:>2}:{idx + dim:<2}] {name} (dim={dim})")
            idx += dim
        print(f"  [{idx:>2}:{idx + self.action_dim:<2}] prev_actions (dim={self.action_dim})")
        print(f"[ObservationBuilder] force_threshold={self.force_threshold}, "
              f"obs_dim={self.obs_dim}")

    def validate_against_checkpoint(self, checkpoint_obs_dim: int):
        """Fail loudly if the assembled obs width doesn't match the trained policy."""
        if self.obs_dim != checkpoint_obs_dim:
            raise ValueError(
                f"Observation dimension mismatch: ObservationBuilder produces "
                f"{self.obs_dim} but the checkpoint preprocessor expects "
                f"{checkpoint_obs_dim}. obs_order={self.obs_order}, "
                f"action_dim={self.action_dim}."
            )

    def build_observation(
        self,
        snapshot,
        goal_position: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the raw (pre-normalization) observation vector.

        Args:
            snapshot: ``StateSnapshot`` from the FrankaInterface.
            goal_position: [3] hole-entrance obs-frame position in world coords.
            prev_actions: [action_dim] previous EMA-smoothed action (dims 3:5 are
                          re-zeroed here to match FORGE `_get_observations`).

        Returns:
            [obs_dim] observation tensor (single env, unbatched).
        """
        components = {
            "fingertip_pos_rel_fixed": snapshot.ee_pos - goal_position,
            "fingertip_quat": snapshot.ee_quat.clone(),
            "ee_linvel": snapshot.ee_linvel,
            "ee_angvel": snapshot.ee_angvel,
            "ft_force": snapshot.force_torque[:3].clone(),
            "force_threshold": torch.tensor(
                [self.force_threshold], device=self.device, dtype=torch.float32
            ),
        }

        parts = []
        for name in self.obs_order:
            c = components[name]
            if c.dim() == 0:
                c = c.unsqueeze(0)
            if c.shape[0] != OBS_DIM_MAP[name]:
                raise RuntimeError(
                    f"Observation term '{name}' has dim {c.shape[0]}, "
                    f"expected {OBS_DIM_MAP[name]}"
                )
            parts.append(c.to(self.device, dtype=torch.float32))

        # prev_actions with rotation roll/pitch dims (3:5) zeroed, matching FORGE.
        pa = prev_actions.to(self.device, dtype=torch.float32).clone()
        if pa.shape[0] != self.action_dim:
            raise RuntimeError(
                f"prev_actions has dim {pa.shape[0]}, expected {self.action_dim}"
            )
        pa[3:5] = 0.0
        parts.append(pa)

        obs = torch.cat(parts, dim=0)
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"Assembled obs has dim {obs.shape[0]}, expected {self.obs_dim}"
            )
        return obs


class ObservationNormalizer:
    """Applies the frozen training normalizer (skrl RunningStandardScaler stats).

    Loads ``running_mean`` / ``running_variance`` from the checkpoint's
    ``observation_preprocessor`` state and applies
    ``(obs - mean) / sqrt(var + eps)`` — no updates during evaluation.

    Args:
        preprocessor_state: The per-agent preprocessor state dict from the
            checkpoint (``ckpt["observation_preprocessor"]``).
        device: Torch device.
        eps: Numerical-stability epsilon (skrl default 1e-8).
        obs_dim: If given and smaller than the stored stats, slice to the first
            ``obs_dim`` entries (the policy portion).
    """

    def __init__(self, preprocessor_state: dict, device: str = "cpu",
                 eps: float = 1e-8, obs_dim: int = None):
        self.device = device
        self.eps = eps

        if preprocessor_state is None:
            raise ValueError("Checkpoint has no 'observation_preprocessor' state")
        if "running_mean" not in preprocessor_state:
            raise ValueError("observation_preprocessor missing 'running_mean'")
        if "running_variance" not in preprocessor_state:
            raise ValueError("observation_preprocessor missing 'running_variance'")

        mean = preprocessor_state["running_mean"].to(device).float().flatten()
        var = preprocessor_state["running_variance"].to(device).float().flatten()
        full = mean.shape[0]

        if obs_dim is not None and obs_dim < full:
            mean, var = mean[:obs_dim], var[:obs_dim]
        elif obs_dim is not None and obs_dim > full:
            raise ValueError(
                f"obs_dim={obs_dim} exceeds preprocessor dim={full}; "
                f"check obs_order reconstruction."
            )

        self.running_mean = mean
        self.running_variance = var
        self.obs_dim = self.running_mean.shape[0]

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.running_mean) / torch.sqrt(self.running_variance + self.eps)
