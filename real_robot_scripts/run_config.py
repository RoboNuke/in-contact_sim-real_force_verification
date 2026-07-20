"""
Resolve a training run's `runtime_config.yaml` and derive its wandb identity.

The runner dumps the full ConfigManager config and `MetricWriter` uploads it to
each wandb run's Files as `runtime_config.yaml` (learning/metric_writer.py). It is
the authoritative source of every training-derived value the real eval needs that
varies per run: model architecture (`model_cfg`), `sac_cfg.predict_success`, the
breakable-peg `break_force`, and the force-obs mode (`force_obs_cfg`).

This module resolves that config from either wandb (download by run id) or the
local run directory, and derives the wandb project/group/tags the training run
used so the real-robot eval can be logged as a sibling run (same project/group/
tags + a `real_robot_eval` marker).

FORGE env constants (obs_order, action bounds, ema, force_threshold, dof pose) are
NOT in this config — see forge_defaults.py.
"""

import glob
import os
import re

from configs.manager import ConfigManager

REAL_EVAL_TAG = "real_robot_eval"


def _local_run_dir(checkpoint_dir: str, agent_idx: int) -> str:
    """Return the `<checkpoint>/<agent>` (or `<checkpoint>`) dir that holds the run."""
    cand = os.path.join(checkpoint_dir, str(agent_idx))
    return cand if os.path.isdir(cand) else checkpoint_dir


def discover_local_runtime_config(checkpoint_dir: str, agent_idx: int):
    """Find (runtime_config_path, run_id) under a local run dir's wandb files.

    Returns (path, run_id) or (None, None). run_id is parsed from the
    `wandb/run-<ts>-<id>` directory name.
    """
    base = _local_run_dir(checkpoint_dir, agent_idx)
    matches = sorted(glob.glob(os.path.join(base, "wandb", "run-*", "files", "runtime_config.yaml")))
    if not matches:
        return None, None
    path = matches[-1]
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))  # run-<ts>-<id>
    m = re.match(r"run-\d+_\d+-(?P<id>[a-z0-9]+)$", run_dir)
    run_id = m.group("id") if m else None
    return path, run_id


def download_runtime_config_from_wandb(entity: str, project: str, run_id: str, cache_dir: str) -> str:
    """Download `runtime_config.yaml` from a wandb run's Files. Returns local path."""
    import wandb
    if not (entity and project and run_id):
        raise ValueError(
            "wandb download needs --wandb_entity, --wandb_project, and --wandb_run"
        )
    os.makedirs(cache_dir, exist_ok=True)
    api = wandb.Api(timeout=60)
    run = api.run(f"{entity}/{project}/{run_id}")
    dst = os.path.join(cache_dir, f"{run_id}_runtime_config.yaml")
    f = run.file("runtime_config.yaml")
    f.download(root=cache_dir, replace=True)
    downloaded = os.path.join(cache_dir, "runtime_config.yaml")
    if os.path.exists(downloaded) and downloaded != dst:
        os.replace(downloaded, dst)
    print(f"[run_config] downloaded runtime_config.yaml from wandb {entity}/{project}/{run_id}")
    return dst


def resolve_runtime_config(
    checkpoint_dir: str,
    agent_idx: int,
    wandb_run: str = None,
    wandb_entity: str = None,
    wandb_project: str = None,
    cache_dir: str = "./runtime_config_cache",
):
    """Resolve the training config for a checkpoint.

    Priority:
      1. `--wandb_run` given -> download runtime_config.yaml from wandb.
      2. local `<run>/<agent>/wandb/run-*/files/runtime_config.yaml`.
      3. local `<run>/<agent>/config.yaml` (the ConfigManager dump).

    Returns (config_dict, source_path, run_id).
    """
    run_id = wandb_run
    if wandb_run:
        path = download_runtime_config_from_wandb(wandb_entity, wandb_project, wandb_run, cache_dir)
    else:
        path, run_id = discover_local_runtime_config(checkpoint_dir, agent_idx)
        if path is None:
            base = _local_run_dir(checkpoint_dir, agent_idx)
            fallback = os.path.join(base, "config.yaml")
            if not os.path.isfile(fallback):
                raise FileNotFoundError(
                    f"No runtime_config.yaml (wandb files) or config.yaml found under {base}. "
                    f"Pass --wandb_run/--wandb_entity/--wandb_project to download from wandb."
                )
            path = fallback

    print(f"[run_config] training config: {path}")
    cfg = ConfigManager.load(path)
    return cfg, path, run_id


def derive_wandb_target(cfg: dict, agent_idx: int, extra_tags=None,
                        project_override: str = None, entity_override: str = None) -> dict:
    """Derive the eval run's wandb identity from the training config.

    Reproduces learning/metric_writer.py::make_wandb_run so the eval lands in the
    SAME project/group with the SAME tags as training, plus a `real_robot_eval`
    tag. Overrides win when provided.
    """
    exp = cfg["sac_cfg"].experiment
    wk = dict(getattr(exp, "wandb_kwargs", {}) or {})

    directory = str(getattr(exp, "directory", "") or "")
    exp_name = getattr(exp, "experiment_name", "") or os.path.basename(directory.rstrip("/")) or "experiment"
    for ext in (".yaml", ".yml"):
        if exp_name.lower().endswith(ext):
            exp_name = exp_name[: -len(ext)]
    family = os.path.basename(os.path.dirname(directory.rstrip("/"))) or "skrl"

    project = project_override or wk.get("project") or family
    entity = entity_override or wk.get("entity")
    group = wk.get("group") or exp_name

    tags = list(wk.get("tags", []) or [])
    for t in (list(extra_tags or []) + [REAL_EVAL_TAG]):
        if t not in tags:
            tags.append(t)

    return {
        "project": project,
        "entity": entity,
        "group": group,
        "tags": tags,
        "name": f"{exp_name}_agent{agent_idx}_realeval",
    }
