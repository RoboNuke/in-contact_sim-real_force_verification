"""Sweep the force-control gain and report which tracks the target force best.

Boots Isaac once, builds the force-mode env, and for each gain: resets (re-grasp
+ place), settles, then measures the contact and joint-torque-estimate force over
a window (mean +/- std, so oscillation shows up as large std). Lower |mean-target|
and lower std => better tracking.

Example:
    python scripts/sweep_force_gain.py --num_envs 4 --headless --force_target 10
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep the force-control gain.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--force_target", type=float, default=10.0, help="z force target (N).")
parser.add_argument("--settle", type=int, default=60, help="policy steps to settle before measuring.")
parser.add_argument("--measure", type=int, default=40, help="policy steps to average over.")
parser.add_argument(
    "--gains",
    type=float,
    nargs="+",
    default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    help="force gains to sweep.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import envs  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg("Isaac-ContactForceTest-Direct-v0", device=args.device, num_envs=args.num_envs)
    env_cfg.ctrl.control_mode = "force"
    env = gym.make("Isaac-ContactForceTest-Direct-v0", cfg=env_cfg, render_mode=None)
    u = env.unwrapped

    print(
        f"[sweep] force_target={args.force_target} N  num_envs={args.num_envs}  "
        f"settle={args.settle}  measure={args.measure}  clamp={u.cfg.ctrl.force_wrench_bound} N",
        flush=True,
    )
    print(f"[sweep] {'gain':>5} | {'contact mean±std':>18} | {'est mean±std':>18} | {'|est-tgt|':>9}", flush=True)

    best = None
    for gain in args.gains:
        env.reset()
        action = torch.tensor([[args.force_target, gain]], device=u.device).repeat(args.num_envs, 1)
        for _ in range(args.settle):
            env.step(action)
        contacts, ests = [], []
        for _ in range(args.measure):
            env.step(action)
            contacts.append(u.contact_force_ee[:, 2].mean().item())
            ests.append(u.est_force_ee_meas[:, 2].mean().item())
        c = torch.tensor(contacts)
        e = torch.tensor(ests)
        c_m, c_s, e_m, e_s = c.mean().item(), c.std().item(), e.mean().item(), e.std().item()
        err = abs(e_m - args.force_target)
        # tracking score: closeness to target + oscillation penalty (lower is better).
        score = err + e_s
        if best is None or score < best[1]:
            best = (gain, score)
        print(
            f"[sweep] {gain:5.1f} | {c_m:8.2f} ± {c_s:6.2f} | {e_m:8.2f} ± {e_s:6.2f} | {err:9.2f}",
            flush=True,
        )

    print(f"[sweep] best-tracking gain (min |est-tgt| + std): {best[0]:.1f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
