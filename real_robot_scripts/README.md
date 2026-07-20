# Real-robot contact-force test (FR3)

Hardware counterpart of the sim contact-force push sweep
(`scripts/param_sweep_run.py` + `envs/contact_force_test_env.py`). It runs the
identical protocol on a real Franka FR3 and logs the same data, so sim and real
force readings can be compared directly.

The FR3 interface (`FrankaInterface`, the hybrid controller, quaternion/frame
utilities, and a mock robot) is **vendored** here from RoboNuke's
`Continuous_Force_RL/real_robot_exps/` and customized — nothing at runtime
depends on that repo.

## Layout

| file | purpose |
|------|---------|
| `contact_force_test.py` | the experiment driver (run this) |
| `config.yaml` | robot connection, frames, F/T filtering, and the experiment protocol/gains |
| `pro_robot_interface.py` | process-based `FrankaInterface` (1 kHz comm + compute in separate processes) |
| `robot_interface.py` | `StateSnapshot`, safety, quaternion/frame helpers, FR3 limits |
| `hybrid_controller.py` | `ControlTargets` + pure-PyTorch wrench→torque control math |
| `mock_pylibfranka.py` | fake robot for off-hardware dry runs (`--mock`) |

## What it does

The **surface reference** (the EE/fingertip position at which the peg tip
contacts the surface) comes from `surface_pos` in the config, or is captured
from a hand-set pose when `surface_pos: null`. The orientation is leveled
**exactly upright** (tool axis straight down), so the push is purely vertical
regardless of small tilts in the pose. Then, for each target force and each
repeat — mirroring the sim reset — the script:

1. centers `approach_height_m` (default **2 cm**) above the surface point,
2. re-tares the F/T estimate in that free space (`calibrate_ft_bias`),
3. lowers to `tip_gap_m` above the surface (the sim's **0.2 mm** `tip_gap`),
4. switches to torque **position control** and holds the standoff pose still for
   `settle_seconds` (default **0.5 s**, unlogged) so the mode-switch transient
   settles, then
5. pushes straight down to `z = surface_ref_z − force/gain`, holding
   `hold_seconds` (2 s) while logging, then returns to (1) for the next rep.

`gain` is the z proportional gain (default **565 N/m** — the sim
`default_task_prop_gains[2]`). The surface is rigid, so the controller settles at
`force = gain × depth`; the "stiffness" that sets the force is the controller
gain, which is known exactly. Defaults: forces `[1, 2, 5, 10, 15]` N × 10 reps =
50 pushes, 2 s each.

## Setting the surface reference

**Preferred — fix it in the config.** Set `surface_pos: [x, y, z]` (world-frame
EE position where the peg tip touches the surface). Every push then stages above
it and lowers to the `tip_gap_m` standoff, with no hand-positioning between runs.
To find the value: run once with `surface_pos: null` (procedure below) and paste
the printed `[start] surface_pos` into the config.

**Or capture it by hand** (`surface_pos: null`):

1. Put the peg in the gripper (the script clamps it with `close_gripper`; use
   `--no_grip` for a fixed tool).
2. Jog/hand-guide the robot so the peg tip is **just touching the surface** at
   the spot you want to test (orientation is auto-leveled upright, so exact roll
   isn't critical, but keep it near vertical). Place the z carefully: at 1 N the
   commanded depth is only ~1.8 mm, so a 0.2 mm reference error is ~11 % of the
   target.
3. Run the script. It reads this position once at start; every rep then stages
   `approach_height_m` above it and lowers to the `tip_gap_m` standoff before
   pushing.

Leveling can be disabled with `force_upright: false` (or `--no_upright`) to hold
the pose orientation as-is; a tilt larger than `upright_warn_deg` prints a
warning.

## Running

Dry run (no hardware — validates the full loop, process spawn, logging, save):

```bash
python real_robot_scripts/contact_force_test.py \
    --config real_robot_scripts/config.yaml --mock \
    --forces 1,2 --reps 2 --hold 0.5
```

On the robot (set `robot.ip` and `use_mock: false` in `config.yaml` first):

```bash
python real_robot_scripts/contact_force_test.py --config real_robot_scripts/config.yaml
```

Useful overrides: `--forces 1,2,5,10,15`, `--reps N`, `--hold SECONDS`,
`--out_dir DIR`, `--no_grip`. **First on-robot run: start small**
(`--forces 1 --reps 2`) to confirm force levels and stability.

## Output

One compressed `.npz` in `out_dir` (default `data/real_robot/`),
`contact_force_real_<timestamp>.npz`.

**15 Hz snapshot fields**, stacked to shape `(n_forces, n_reps, n_steps, …)` —
these mirror the sim per-step log (`envs/contact_force_test_env.py`):

| field | shape tail | sim counterpart |
|-------|-----------|-----------------|
| `joint_pos`, `joint_vel` | `(7,)` | same (arm joints) |
| `joint_torques_meas` | `(7,)` | `joint_torque_measured` (measured `tau_J`) |
| `mass_matrix` | `(7,7)` | `mass_matrix` |
| `jacobian` | `(6,7)` | `jacobian` |
| `ee_pos`, `ee_quat`, `ee_linvel`, `ee_angvel` | `(3/4,)` | same |
| `ft_ee` | `(6,)` | `contact_force_ee` — real analog (from `O_F_ext_hat_K`; the FR3 has no true contact sensor) |
| `est_force_ee_pinv` | `(3,)` | `est_force_ee_pinv` — `pinv(Jᵀ)·τ`, EE frame |
| `est_force_ee_dyn` | `(3,)` | `est_force_ee_dynconsistent` — `J̄ᵀ·τ`, EE frame |

`est_force_ee_pinv/dyn` are recomputed here with the same formulas as the sim
(`_map_torque_to_ee_force[_dyn]`), from the measured `tau_J`, Jacobian, and mass
matrix.

**1 kHz trajectory:** `traj_1khz`, an object array `(n_forces, n_reps)`; each
cell is a dict of the built-in high-rate buffer: `time_ms`, `ee_pos`, `ee_quat`,
`joint_pos`, `joint_vel`, `ft_raw`, `ft_filtered`, `joint_torques_cmd`
(commanded/applied-torque analog), `task_wrench`. Load with
`np.load(path, allow_pickle=True)`.

**Metadata:** `force_targets`, `depths`, `tare_bias (n_forces,n_reps,6)`,
`meta_gain`, `prop_gains`, `control_rate_hz`, `hold_seconds`, `n_steps`,
`settle_seconds`, `settle_steps`, `reps`,
`surface_pos` (= `start_ee_pos`), `start_ee_quat` (leveled) / `start_ee_quat_raw`
(as captured), `start_joint_q`, `surface_ref_z`, `approach_height_m`,
`tip_gap_m`, `wall_time_total`, `use_mock`.

## Safety / notes

- The interface caps force at 50 N (`_SAFETY_MARGIN_FORCE`), slews torque at
  ≤ 0.75 Nm/ms, and `check_safety` runs every 15 Hz step (joint pos/vel/force
  limits, process health). 15 N is well within limits.
- **Gain 565 is stiff for the FR3.** It's the default for sim fidelity, but
  RoboNuke's tuned real gains are ~100–150. If the push feels unstable, lower
  `experiment.prop_gains` / `experiment.gain` in `config.yaml` (note: changing
  `gain` also rescales the depth-per-force, keeping `force = gain × depth`).
- Ctrl-C or any safety violation stops the robot on the next comm timeout;
  `shutdown()` always runs.


---

# Real-robot FORGE peg-insert policy evaluation

Runs a locally-trained SAC **FORGE peg-insert** (breakable-peg) policy on the real
FR3 for a batch of insertion episodes, logs aggregate metrics to wandb, and
(optionally) records per-step data + a video for every trial. Drives the arm from
the **policy** each 15 Hz step via the vendored `FrankaInterface` + the tested
`ControlTargets` pose path.

## Where each value comes from

| source | values |
|--------|--------|
| **run's `runtime_config.yaml`** (wandb download or local run dir) | model architecture, `predict_success`, **`break_force`** (e.g. 10 N), force-obs mode |
| **`forge_defaults.py`** (FORGE-fixed, same every run) | obs_order, action bounds/thresholds, action EMA, `force_threshold`, default joint pose |
| **`eval_config.yaml`** (real-robot-unique only) | robot connection, calibrated goal, hardware PD gains, reset noise, camera, wandb target |

So `break_force` is never hand-set in the eval — it is read from the checkpoint's
own training config (pass `--wandb_run <id> --wandb_entity <e> --wandb_project <p>`
to fetch it from wandb, or it is read from `<run>/<agent>/wandb/.../runtime_config.yaml`).

## Files

| file | purpose |
|------|---------|
| `peg_insert_eval.py` | eval driver (episodes, detection, wandb, optional per-step data/video) |
| `eval_config.yaml` | real-robot-unique config only |
| `calibrate_goal.py` | capture the hole goal from a hand-guided pose -> write into `eval_config.yaml` |
| `run_config.py` | resolve `runtime_config.yaml` (wandb/local); derive the wandb project/group/tags |
| `forge_defaults.py` | FORGE task constants (obs_order, action bounds, ema, force_threshold, dof pose) |
| `observation_builder.py` / `forge_policy.py` / `forge_action_map.py` | obs assembly / policy load / action->pose map |
| `camera.py` | RealSense color capture (background thread) + mock fallback |
| `trial_data.py` | per-step DataFrame + mp4 (imageio) + frame<->step matching (local only) |
| `run_peg_insert_eval.sh` | step-by-step runner |

## wandb

The eval is logged as a **sibling of the training run**: same `project` / `group` /
`tags`, plus a `real_robot_eval` tag, named `<exp>_agent<i>_realeval`. Only aggregate
`Eval_Core/*` metrics are uploaded — **per-step data and videos stay local**. Set
`wandb.project` / `wandb.entity` in `eval_config.yaml` to override, or `--no_wandb`.

## Per-step data + video (`--with_step_data`)

One flag turns on both comprehensive per-step logging and RealSense capture. After
each trial (off the control loop) it writes, under
`data_dir/<run>_a<agent>_s<step>/`:

- `ep_<i>.pkl` — a pandas DataFrame with **all** per-step signals (ee pose/vel,
  full F/T, joint pos/vel/torque, raw + EMA actions, target pose, peg-base/xy/z,
  the 24-D obs, success/break flags, `t_mono`) plus the matched `frame_idx`.
- `ep_<i>.mp4` — the trial video (imageio/ffmpeg), with step/force/outcome telemetry
  burned in (`camera.overlay`).
- `ep_<i>.csv` (scalar columns) and `summary.pkl/.csv` (one row per episode).

Frames are matched to policy steps by nearest `time.monotonic()` timestamp, so the
video and DataFrame are aligned both ways.

## Environment

On-robot the eval imports `skrl` + `pylibfranka` (+ `pandas` + `pyrealsense2` for
`--with_step_data`). No stock env has all of these — set up one and point `EVAL_ENV`
at it:

```bash
conda activate libfranka-python && pip install skrl pandas pyrealsense2
export EVAL_ENV=libfranka-python
```

The **mock dry-run** needs only `skrl`+`torch`+`pandas`+`cv2` (the training env
`fail` has these; the camera falls back to a synthetic MockRecorder off-hardware).

## Step-by-step

`<ckpt>` = a training run dir (e.g. `runs/viability_test/contact_baseline`).

```bash
# 0. Off-hardware dry-run: checkpoint load, obs-dim match, full loop, action map,
#    metrics, goal write-back. No robot, no wandb.
real_robot_scripts/run_peg_insert_eval.sh dry <ckpt>

# 1. Calibrate the goal (on robot): close gripper, hand-guide the peg until FULLY
#    SEATED, release guiding, press Enter. Writes fixed_asset_position +
#    target_peg_base_position into eval_config.yaml, then verifies alignment.
real_robot_scripts/run_peg_insert_eval.sh calibrate <ckpt>

# 2. (optional) Cross-check the assembled observation against a known live pose.
real_robot_scripts/run_peg_insert_eval.sh readstate <ckpt>

# 3. Smoke eval: 2 episodes, no wandb. Watch max_force vs the training break_force.
real_robot_scripts/run_peg_insert_eval.sh smoke <ckpt>

# 4. Full eval -> wandb (sibling of the training run). Add --with_step_data to also
#    save per-step DataFrames + videos locally.
AGENT=0 NUM_EPISODES=20 real_robot_scripts/run_peg_insert_eval.sh full <ckpt> --with_step_data
```

`peg_insert_eval.py` flags: `--checkpoint <run> --agent <i> --step <n>` (default
latest), `--num_episodes`, `--with_step_data`, `--data_dir`, `--std_scale`,
`--no_wandb`, `--wandb_run/--wandb_entity/--wandb_project` (download runtime_config
from wandb), `--override key=value`.

## Safety / notes

- **Start with `smoke`.** Watch `max_force` vs the training `break_force` (printed at
  startup, e.g. 10 N). Gains in `eval_config.yaml` (`control.task_prop_gains`) are
  tunable without touching the checkpoint; per-step position moves are delta-clipped
  to 2 cm.
- Success/break detection is a real-world geometric/force proxy — tune
  `task.xy_centering_threshold` / `success_threshold` / `ee_to_peg_base_offset`.
- The interface caps force at 50 N, slews torque ≤ 0.75 Nm/ms, runs `check_safety`
  each step; Ctrl-C / any safety violation stops the robot and `shutdown()` runs.
