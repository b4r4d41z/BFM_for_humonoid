# Test_training/scripts/bc/play.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

from cli_args import build_parser


def _add_extension_to_syspath() -> Path:
    """
    Make sure we can import:
      Test_training/ (the IsaacLab extension python package)
    without requiring editable install.
    """
    this_dir = Path(__file__).resolve().parent                 # .../Test_training/scripts/bc
    test_training_dir = this_dir.parent.parent                 # .../Test_training
    py_pkg_root = test_training_dir / "source" / "Test_training"  # contains Test_training/ package
    sys.path.insert(0, str(py_pkg_root))
    return py_pkg_root


def _launch_sim_app(args) -> Any:
    """
    Launch Isaac Sim / IsaacLab app runtime (provides `omni`).
    Must be done BEFORE importing isaaclab envs/tasks.
    """
    try:
        from isaaclab.app import AppLauncher  # type: ignore
    except Exception:
        # fallback for some Isaac distributions
        from omni.isaac.lab.app import AppLauncher  # type: ignore

    app_launcher = AppLauncher(args)
    return app_launcher.app


def make_env(args) -> Any:
    """
    Create IsaacLab env. Called only AFTER simulation app is launched.
    """
    import gymnasium as gym

    # IsaacLab utility that builds cfg registered for a task string
    try:
        from isaaclab_tasks.utils import parse_env_cfg  # type: ignore
    except Exception:
        # fallback path for some installs
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    return env


def _infer_act_dim(env_raw: Any) -> int:
    if hasattr(env_raw, "action_space"):
        try:
            shape = getattr(env_raw.action_space, "shape", None)
            if shape is not None and len(shape) >= 1:
                return int(shape[0])
        except Exception:
            pass
    raise RuntimeError(
        "Cannot infer action dimension from env.action_space. "
        "Please set act_dim explicitly for your env."
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # 1) Launch Isaac runtime FIRST (creates `omni`)
    simulation_app = _launch_sim_app(args)

    # 2) Only now add your extension to sys.path and import Test_training.*
    _add_extension_to_syspath()

    # These imports would previously crash because they trigger `Test_training/__init__.py -> tasks -> omni`
    from bc.wrappers.isaaclab_env import IsaacLabEnvWrapper
    from bc.fb_cpr.agent import FBCPRAgent, AgentConfig
    from bc.fb_cpr.model import ModelConfig

    # 3) Create env after runtime is up
    env_raw = make_env(args)
    env = IsaacLabEnvWrapper(env_raw, obs_key=args.obs_key, flatten_dict=True)

    obs, info = env.reset(seed=args.seed)
    print(f"[BFM play] device={env.device} num_envs={env.num_envs} obs_shape={tuple(obs.shape)}")

    obs_dim = int(obs.shape[1])
    act_dim = _infer_act_dim(env_raw)

    agent: Optional[FBCPRAgent] = None
    if args.mode == "agent":
        model_cfg = ModelConfig(obs_dim=obs_dim, act_dim=act_dim)
        agent_cfg = AgentConfig(device=str(env.device) if args.device is None else args.device)
        agent = FBCPRAgent(model_cfg, agent_cfg)
        if args.checkpoint:
            agent.load(args.checkpoint)

    total_rew = 0.0
    done_count = 0

    for t in range(1, int(args.steps) + 1):
        if args.mode == "random":
            action = torch.randn((env.num_envs, act_dim), device=env.device, dtype=torch.float32)
            action = torch.tanh(action)
        else:
            assert agent is not None
            action = agent.act(obs, deterministic=False).to(env.device)

        out = env.step(action)
        obs = out.obs

        total_rew += float(out.reward.mean().item())
        done_count += int(out.done.sum().item())

        if t % int(args.print_every) == 0:
            avg_rew = total_rew / float(t)
            print(f"[BFM play] step={t} avg_reward={avg_rew:.4f} done_count={done_count}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
