"""
FORGE environment constants shared by every FORGE peg-insert run.

These live in Isaac Lab (`direct/forge/forge_env_cfg.py`, `direct/factory/
factory_env_cfg.py`), not in our ConfigManager YAML, so they are identical across
training runs and are NOT part of `runtime_config.yaml`. They are also not unique
to the real robot — they define the FORGE task's obs/action semantics — so they
belong in code, not in `eval_config.yaml`. The eval reproduces them here.

Real-robot-unique values (connection, calibrated goal, hardware-tuned PD gains,
reset noise, camera, wandb target) stay in `eval_config.yaml`. Training-derived
values that DO vary per run (break_force, model architecture, force-obs mode) come
from the run's `runtime_config.yaml` (see run_config.py).
"""

# Policy observation term order (forge_env_cfg.py obs_order). prev_actions(7) is
# appended after these by the env / ObservationBuilder.
OBS_ORDER = [
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "ft_force",
    "force_threshold",
]

ACTION_DIM = 7

# Action-mapping constants (factory CtrlCfg / forge_env _apply_action).
POS_ACTION_BOUNDS = [0.05, 0.05, 0.05]      # position action scale (m)
ROT_ACTION_BOUNDS = [1.0, 1.0, 1.0]         # rotation action scale
POS_ACTION_THRESHOLD = [0.02, 0.02, 0.02]   # per-step position delta clip (m)
ROT_ACTION_THRESHOLD = [0.097, 0.097, 0.097]  # per-step euler delta clip (rad)

# FORGE randomizes the action EMA per episode over this range; eval uses a fixed
# value (the midpoint) for determinism.
EMA_FACTOR_RANGE = [0.025, 0.1]
EMA_FACTOR_DEFAULT = 0.0625

# FORGE randomizes the contact-penalty threshold (the `force_threshold` obs
# channel) per episode over this range; eval feeds the fixed midpoint.
CONTACT_PENALTY_THRESHOLD_RANGE = [5.0, 10.0]
FORCE_THRESHOLD_DEFAULT = 7.5

# Factory default arm joint configuration (null-space target).
DEFAULT_DOF_POS = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
