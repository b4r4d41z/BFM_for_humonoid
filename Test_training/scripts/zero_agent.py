# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time
import threading

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# NEW: diagnostics
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Max number of env steps. 0 means run until the app is closed.",
)
parser.add_argument(
    "--log_every",
    type=int,
    default=200,
    help="Print progress every N steps. 0 disables periodic logging.",
)
parser.add_argument(
    "--watchdog_sec",
    type=float,
    default=10.0,
    help="Warn if no progress for this many seconds (best-effort). Set 0 to disable.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Test_training.tasks  # noqa: F401


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    env.reset()

    # diagnostics state
    step = 0
    t_start = time.time()
    t_last_log = time.time()
    last_progress = {"t": time.time()}
    stop_event = threading.Event()

    def watchdog():
        """Best-effort watchdog: warns if steps stop completing for too long."""
        if args_cli.watchdog_sec <= 0:
            return
        while not stop_event.wait(timeout=max(0.5, args_cli.watchdog_sec / 2.0)):
            dt = time.time() - last_progress["t"]
            if dt > args_cli.watchdog_sec:
                print(f"[WARN]: No progress for {dt:.1f}s (last completed step: {step}).")

    wd_thread = threading.Thread(target=watchdog, daemon=True)
    wd_thread.start()

    try:
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # compute zero actions
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                # apply actions
                _, reward, terminated, truncated, _ = env.step(actions)

            step += 1
            last_progress["t"] = time.time()

            # periodic progress log
            if args_cli.log_every > 0 and (step % args_cli.log_every == 0):
                now = time.time()
                dt_block = now - t_last_log
                sps = (args_cli.log_every / dt_block) if dt_block > 0 else float("inf")
                t_last_log = now
                elapsed = now - t_start

                # Best-effort reward summary
                r_mean = None
                try:
                    if isinstance(reward, torch.Tensor):
                        r_mean = float(reward.mean().item())
                    else:
                        r_mean = float(sum(reward) / len(reward))
                except Exception:
                    r_mean = None

                # Best-effort done summary
                done_any = False
                try:
                    if isinstance(terminated, torch.Tensor):
                        done_any = bool(terminated.any().item()) or bool(truncated.any().item())
                    else:
                        done_any = any(terminated) or any(truncated)
                except Exception:
                    done_any = False

                if r_mean is not None:
                    print(
                        f"[INFO]: step={step} elapsed={elapsed:.1f}s sps={sps:.1f} "
                        f"reward_mean={r_mean:.4f} done_any={done_any}"
                    )
                else:
                    print(f"[INFO]: step={step} elapsed={elapsed:.1f}s sps={sps:.1f} done_any={done_any}")

            # hard step limit
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                print(f"[INFO]: Reached max_steps={args_cli.max_steps}. Exiting.")
                break

    except KeyboardInterrupt:
        print("[INFO]: Interrupted by user (Ctrl+C).")
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
