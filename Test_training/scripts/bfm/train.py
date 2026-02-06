from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_THIS_DIR = Path(__file__).resolve().parent
_TEST_TRAINING_DIR = _THIS_DIR.parent.parent
_PY_PKG_ROOT = _TEST_TRAINING_DIR / "source" / "Test_training"
sys.path.insert(0, str(_PY_PKG_ROOT))

from Test_training.learning.bfm.wrappers.isaaclab_env import IsaacLabEnvWrapper  # noqa: E402
from Test_training.learning.bfm.buffers.buffers import ReplayBuffer  # noqa: E402
from Test_training.learning.bfm.fb_cpr.agent import FBCPRAgent, AgentConfig  # noqa: E402
from Test_training.learning.bfm.fb_cpr.model import ModelConfig  # noqa: E402

from .cli_args import build_parser  # noqa: E402


def make_env(args) -> Any:
    import gymnasium as gym
    try:
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore
    except Exception:
        from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    return env


def main() -> None:
    parser = build_parser()
    # extra train args
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--update_every", type=int, default=2000)
    parser.add_argument("--save_path", type=str, default="bfm_ckpt.pt")
    args = parser.parse_args()

    simulation_app = None
    try:
        from omni.isaac.lab.app import AppLauncher  # type: ignore

        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
    except Exception:
        pass

    env_raw = make_env(args)
    env = IsaacLabEnvWrapper(env_raw, obs_key=args.obs_key, flatten_dict=True)

    obs, _ = env.reset(seed=args.seed)
    obs_dim = obs.shape[1]

    if not hasattr(env_raw, "action_space"):
        raise RuntimeError("Env has no action_space; cannot infer act_dim.")
    act_dim = int(env_raw.action_space.shape[0])

    model_cfg = ModelConfig(obs_dim=obs_dim, act_dim=act_dim)
    agent_cfg = AgentConfig(device=args.device)
    agent = FBCPRAgent(model_cfg, agent_cfg)
    if args.checkpoint:
        agent.load(args.checkpoint)

    buffer = ReplayBuffer(
        capacity=int(args.buffer_size),
        device=torch.device(args.device),
        obs_dim=obs_dim,
        act_dim=act_dim,
    )

    total_rew = 0.0
    for t in range(1, int(args.steps) + 1):
        act = agent.act(obs, deterministic=False).to(env.device)
        out = env.step(act)

        # store on agent device
        buffer.add(
            obs=obs.to(agent.device),
            act=act.to(agent.device),
            rew=out.reward.to(agent.device),
            done=out.done.to(agent.device),
            next_obs=out.obs.to(agent.device),
        )

        obs = out.obs
        total_rew += float(out.reward.mean().item())

        if t % int(args.print_every) == 0:
            avg_rew = total_rew / float(t)
            print(f"[BFM train] step={t} avg_reward={avg_rew:.4f} buffer_size={len(buffer)}")

        if len(buffer) >= int(args.batch_size) and (t % int(args.update_every) == 0):
            batch = buffer.sample(int(args.batch_size))
            stats = agent.update(batch)
            print(f"[BFM train] update at step={t} stats={stats}")

    agent.save(args.save_path)
    env.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()