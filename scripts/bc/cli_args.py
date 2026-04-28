from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("BFM runner (minimal)")

    # BFM runner args
    parser.add_argument("--task", type=str, required=True, help="IsaacLab task name, e.g. Template-Test-Training-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--steps", type=int, default=200, help="Number of env steps")
    parser.add_argument("--obs_key", type=str, default=None, help="Key in dict obs, e.g. 'policy'")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "agent"])
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=50)

    # Try to attach AppLauncher args if IsaacLab is installed
    try:
        from omni.isaac.lab.app import AppLauncher  # type: ignore

        AppLauncher.add_app_launcher_args(parser)
    except Exception:
        # allow running without IsaacLab import at parse-time
        pass

    return parser
