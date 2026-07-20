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

Runs a locally-trained SAC **FORGE peg-insert** policy (the breakable-peg task) on
the real FR3 for a batch of insertion episodes and logs metrics to wandb. Unlike
the contact-force push test above, this drives the arm from the **policy** each
15 Hz step, using the same vendored `FrankaInterface` and the tested
`ControlTargets` pose path (`sel_matrix=0` ⇒ pure task-space PD).

## Files

| file | purpose |
|------|---------|
| `eval_config.yaml` | robot connection, calibrated goal, control gains, obs constants, reset/noise, wandb |
| `calibrate_goal.py` | capture the hole goal from a hand-guided pose and write it into `eval_config.yaml` |
| `peg_insert_eval.py` | the eval driver (loads a checkpoint, runs episodes, logs `Eval_Core/*`) |
| `observation_builder.py` | assembles the 24-D FORGE policy observation from a state snapshot |
| `forge_policy.py` | loads a local `ckpt_<step>.pt` into a 1-agent `BlockSimBaActor` (deterministic action) |
| `forge_action_map.py` | replicates FORGE `_apply_action` (action → EE pose target, EMA + clipping) |
| `run_peg_insert_eval.sh` | step-by-step runner: `dry` / `calibrate` / `readstate` / `smoke` / `full` |

## What it reproduces from sim

- **Obs (24-D):** `fingertip_pos_rel_fixed(3), fingertip_quat(4), ee_linvel(3),
  ee_angvel(3), ft_force(3), force_threshold(1)` + `prev_actions(7)` (EMA-smoothed,
  dims 3:5 zeroed) — matching `forge_env.py::_get_observations`. `ft_force` is the
  measured contact force; `force_threshold` is the fixed contact-penalty threshold
  (`obs.force_threshold`, sim range [5,10] → midpoint 7.5).
- **Action (7-D):** `[pos_delta(3), rot(3; roll/pitch zeroed, yaw only), success_pred(1)]`,
  mapped to an EE pose target relative to the **fixed (hole) frame** with per-step
  delta clipping and action EMA — matching `forge_env.py::_apply_action`. Deterministic
  eval uses `tanh(mean)`.
- **Break:** `‖force‖ ≥ break_force_threshold` (25 N). **Success:** geometric — peg
  base within `xy_centering_threshold` and below the seated target by `hole_height *
  success_threshold`.
- **Force scaling:** `control.mass_weighting: false` (plain J^T) matches the sim
  controller; `true` inflates force by task-space inertia (the known ~5× sim/real gap).

## Environment

The on-robot steps import **both** `skrl` (policy) and `pylibfranka` (FR3 driver).
No stock env has both — add one to your robot/eval env once, e.g.:

```bash
conda activate libfranka-python && pip install skrl      # or: pip install pylibfranka into the training env
export EVAL_ENV=libfranka-python                          # env used by run_peg_insert_eval.sh
```

The **mock dry-run** only needs `skrl`+`torch`, so the training env (`fail`) works for it.

## Step-by-step

Point `<ckpt>` at a training run dir (contains `<agent>/checkpoints/ckpt_*.pt` and
`<agent>/config.yaml`), e.g. `runs/viability_test/contact_baseline`.

```bash
# 0. Off-hardware dry-run — validates checkpoint load, obs-dim match, the full loop,
#    action mapping, metrics, and the goal write-back. No robot, no wandb.
real_robot_scripts/run_peg_insert_eval.sh dry <ckpt>

# 1. Calibrate the goal (on robot): close gripper, hand-guide the peg until FULLY
#    SEATED in the hole, release guiding, press Enter. Writes fixed_asset_position
#    (hole entrance) and target_peg_base_position into eval_config.yaml, then moves
#    5 cm above the hole so you can eyeball alignment. Set task.ee_to_peg_base_offset
#    and task.hole_height for your peg first.
real_robot_scripts/run_peg_insert_eval.sh calibrate <ckpt>

# 2. (optional) Cross-check the assembled observation against a known live pose —
#    sanity-check ft_force sign/frame and fingertip_pos_rel_fixed before running policies.
real_robot_scripts/run_peg_insert_eval.sh readstate <ckpt>

# 3. Smoke eval — 2 episodes, no wandb. Confirm force levels vs break_force and that
#    the motion is stable before scaling up.
real_robot_scripts/run_peg_insert_eval.sh smoke <ckpt>

# 4. Full eval — logs Eval_Core/* to wandb (project forge_pih). Pick the agent slot
#    and episode count as needed.
AGENT=0 NUM_EPISODES=20 real_robot_scripts/run_peg_insert_eval.sh full <ckpt>
```

`peg_insert_eval.py` takes `--checkpoint <run> --agent <i> --step <n>`
(default: latest step), `--num_episodes`, `--std_scale` (>0 = stochastic),
`--no_wandb`, and `--override key=value` (e.g. `robot.use_mock=true`,
`control.task_prop_gains=[80,80,80,30,30,30]`).

## Safety / notes

- **Start small.** The first on-robot run should be `smoke` (2 episodes). Watch
  `max_force` vs `break_force_threshold` (25 N) and the descent stability.
- **Gains are tunable in `eval_config.yaml`** (`control.task_prop_gains`) without
  touching the checkpoint. The sim default `[100,100,100,30,30,30]` may need
  softening for hardware; per-step position moves are already delta-clipped to 2 cm.
- The interface caps force at 50 N, slews torque ≤ 0.75 Nm/ms, and runs
  `check_safety` every step; Ctrl-C / any safety violation stops the robot and
  `shutdown()` always runs.
- Success/break detection is a real-world **geometric/force proxy** (no sim
  keypoints on hardware) — tune `task.xy_centering_threshold` / `success_threshold`
  / `ee_to_peg_base_offset` to your fixture.
