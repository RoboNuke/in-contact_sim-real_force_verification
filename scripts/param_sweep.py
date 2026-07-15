"""Orchestrate the contact-force position-push parameter sweep.

Sweeps the full Cartesian product of four PhysX / sim parameters:

    * solver_position_iteration_count
    * max_depenetration_velocity
    * sim.dt              (specified as its denominator: 120 => dt = 1/120)
    * contact_offset

For EACH combination it launches ``scripts/param_sweep_run.py`` in a fresh
subprocess (each condition needs a fresh sim, since dt / PhysX iteration counts /
per-body spawn props are baked at sim-build time). That worker runs 5 parallel
envs through force goals 1, 2, 5, 10, 15 N -- resetting before each goal and
holding it for 2 s of sim time -- and writes one ``.npz`` per condition into
``data/param_sweep/``.

``sim.dt`` is the OUTERMOST loop, so every other-parameter combination is
completed for one dt before the next dt begins; if you terminate early you get
complete blocks per dt. Conditions whose ``.npz`` already exists are skipped, so
the sweep is resumable.

This script imports NO Isaac modules -- it only shells out. Run it with the same
Python that can import Isaac Lab::

    python scripts/param_sweep.py
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

# --------------------------------------------------------------------------- #
# Sweep grid -- EDIT THESE. Total conditions = product of the four list sizes. #
# --------------------------------------------------------------------------- #
SOLVER_ITERS = [4, 16, 32, 64, 128, 192, 255]          # solver_position_iteration_count
MAX_DEPEN = [0.1, 0.5, 1.0, 5.0, 20.0]                 # max_depenetration_velocity (m/s)
DT_DENOMS = [60, 120, 240, 480]                        # sim.dt = 1/denom  (OUTERMOST loop)
CONTACT_OFFSETS = [0.002, 0.005, 0.01, 0.02]           # contact_offset (m)

# Fixed protocol (passed through to the worker).
NUM_ENVS = 5
SECONDS = 2.0
FORCES = "1,2,5,10,15"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(_PROJECT_ROOT, "scripts", "param_sweep_run.py")

_p = argparse.ArgumentParser(description="Orchestrate the contact-force parameter sweep.")
_p.add_argument("--task", default="Isaac-ContactForceTest-Direct-v0",
                help="Gym id: Isaac-ContactForceTest-Direct-v0 (cylinder/box) or "
                     "Isaac-ForgeContactTest-Direct-v0 (FORGE peg/hole).")
_p.add_argument("--out_dir", default=None,
                help="Output dir for .npz. Default: data/param_sweep_forge for the FORGE "
                     "task, else data/param_sweep.")
_p.add_argument("--config_src", default=None,
                help="Config file snapshotted into out_dir. Default: matches the task.")
ARGS = _p.parse_args()

TASK = ARGS.task
_IS_FORGE = "Forge" in TASK
OUT_DIR = ARGS.out_dir or os.path.join(
    _PROJECT_ROOT, "data", "param_sweep_forge" if _IS_FORGE else "param_sweep")
CONFIG_SRC = ARGS.config_src or os.path.join(
    _PROJECT_ROOT, "envs",
    "forge_contact_test_env_cfg.py" if _IS_FORGE else "contact_force_test_env_cfg.py")


def _fmt(x) -> str:
    return str(x).replace(".", "-")


def _npz_name(iters, max_depen, dt_denom, contact_offset) -> str:
    return f"{iters}_{_fmt(max_depen)}_{dt_denom}_{_fmt(contact_offset)}.npz"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Snapshot the config used for this sweep + a manifest of the grid.
    shutil.copy2(CONFIG_SRC, os.path.join(OUT_DIR, os.path.basename(CONFIG_SRC)))
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": TASK,
        "solver_position_iteration_count": SOLVER_ITERS,
        "max_depenetration_velocity": MAX_DEPEN,
        "sim_dt_denominators": DT_DENOMS,
        "contact_offset": CONTACT_OFFSETS,
        "num_envs": NUM_ENVS,
        "seconds_per_force": SECONDS,
        "forces_N": FORCES,
        "loop_order_outer_to_inner": ["sim.dt", "solver_iters", "max_depen", "contact_offset"],
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "-C", _PROJECT_ROOT, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        manifest["git_commit"] = None
    with open(os.path.join(OUT_DIR, "sweep_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    total = len(DT_DENOMS) * len(SOLVER_ITERS) * len(MAX_DEPEN) * len(CONTACT_OFFSETS)
    print(f"[sweep] task={TASK}  {total} conditions -> {OUT_DIR}", flush=True)

    done = skipped = failed = 0
    idx = 0
    # sim.dt OUTERMOST so each dt block finishes before the next dt starts.
    for dt_denom in DT_DENOMS:
        for iters in SOLVER_ITERS:
            for max_depen in MAX_DEPEN:
                for contact_offset in CONTACT_OFFSETS:
                    idx += 1
                    name = _npz_name(iters, max_depen, dt_denom, contact_offset)
                    fpath = os.path.join(OUT_DIR, name)
                    tag = f"[{idx}/{total}] {name}"
                    if os.path.exists(fpath):
                        print(f"[sweep] SKIP (exists) {tag}", flush=True)
                        skipped += 1
                        continue

                    cmd = [
                        sys.executable, RUNNER, "--headless",
                        "--task", TASK,
                        "--solver_iters", str(iters),
                        "--max_depen", str(max_depen),
                        "--dt_denom", str(dt_denom),
                        "--contact_offset", str(contact_offset),
                        "--num_envs", str(NUM_ENVS),
                        "--seconds", str(SECONDS),
                        "--forces", FORCES,
                        "--out_dir", OUT_DIR,
                    ]
                    print(f"[sweep] RUN  {tag}", flush=True)
                    rc = subprocess.run(cmd).returncode
                    if rc == 0 and os.path.exists(fpath):
                        done += 1
                    else:
                        print(f"[sweep] FAIL (rc={rc}) {tag}", flush=True)
                        failed += 1

    print(f"[sweep] finished: {done} run, {skipped} skipped, {failed} failed, {total} total.", flush=True)


if __name__ == "__main__":
    main()
