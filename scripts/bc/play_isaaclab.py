from __future__ import annotations

import argparse
from typing import Any

import torch

from isaaclab_wrapper import (
    ActionAdapter,
    BCPolicyRunner,
    ObservationAdapter,
    RolloutRecorder,
    create_isaaclab_env,
)
from isaaclab_wrapper.sanity_checks import (
    check_checkpoint_exists,
    check_device,
    check_shape,
    check_tensor_finite,
    print_debug_tensor,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Play trained BC checkpoint in Isaac Lab")

    parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to BC checkpoint")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0", help="Isaac Lab env device")
    parser.add_argument("--model_device", type=str, default="cuda:0", help="Policy model device")
    parser.add_argument("--max_steps", type=int, default=1000)

    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without GUI")
    parser.add_argument("--render", action="store_true", help="Enable rendering (overrides headless)")

    parser.add_argument("--expected_obs_dim", type=int, default=None)
    parser.add_argument("--expected_action_dim", type=int, default=None)

    parser.add_argument("--obs_dim", type=int, default=None, help="Fallback model construction obs dim")
    parser.add_argument("--action_dim", type=int, default=None, help="Fallback model construction action dim")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Fallback MLP hidden dim")
    parser.add_argument("--hidden_layers", type=int, default=2, help="Fallback MLP hidden layers")

    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--clip_actions", action="store_true", default=True)
    parser.add_argument("--no_clip_actions", action="store_false", dest="clip_actions")

    parser.add_argument("--record_rollout", action="store_true")
    parser.add_argument("--rollout_output_dir", type=str, default="runs/bc/isaaclab_rollouts")

    parser.add_argument("--debug", action="store_true")
    return parser


def _unwrap_reset(reset_out: Any) -> tuple[Any, Any]:
    if isinstance(reset_out, tuple):
        if len(reset_out) >= 2:
            return reset_out[0], reset_out[1]
        if len(reset_out) == 1:
            return reset_out[0], {}
    return reset_out, {}


def _unwrap_step(step_out: Any) -> tuple[Any, Any, Any, Any]:
    if isinstance(step_out, tuple):
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated | truncated
            return obs, reward, done, info
        if len(step_out) == 4:
            obs, reward, done, info = step_out
            return obs, reward, done, info

    if hasattr(step_out, "obs") and hasattr(step_out, "reward"):
        obs = step_out.obs
        reward = getattr(step_out, "reward", None)
        done = getattr(step_out, "done", getattr(step_out, "terminated", None))
        info = getattr(step_out, "info", {})
        return obs, reward, done, info

    raise RuntimeError(f"Unsupported env.step return type: {type(step_out).__name__}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    check_checkpoint_exists(args.checkpoint)

    print(f"[play_isaaclab] selected task: {args.task}")
    print(f"[play_isaaclab] checkpoint path: {args.checkpoint}")
    print(f"[play_isaaclab] environment device: {args.device}")
    print(f"[play_isaaclab] model device: {args.model_device}")

    env = None
    sim_app = None
    recorder = RolloutRecorder(enabled=args.record_rollout, output_dir=args.rollout_output_dir)

    try:
        env, sim_app = create_isaaclab_env(
            task=args.task,
            num_envs=args.num_envs,
            device=args.device,
            headless=args.headless,
            render=args.render,
        )

        obs_adapter = ObservationAdapter(
            expected_obs_dim=args.expected_obs_dim,
            device=args.model_device,
            debug=args.debug,
        )
        act_adapter = ActionAdapter(
            expected_action_dim=args.expected_action_dim,
            env_device=args.device,
            action_scale=args.action_scale,
            clip_actions=args.clip_actions,
            debug=args.debug,
        )

        policy_runner = BCPolicyRunner(
            checkpoint_path=args.checkpoint,
            device=args.model_device,
            debug=args.debug,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
        )

        raw_obs, _ = _unwrap_reset(env.reset())
        print(f"[play_isaaclab] raw observation type: {type(raw_obs).__name__}")

        for step in range(int(args.max_steps)):
            model_obs = obs_adapter(raw_obs)
            check_shape("model_obs", model_obs, expected_last_dim=args.expected_obs_dim)
            check_tensor_finite("model_obs", model_obs)
            check_device("model_obs", model_obs, args.model_device)

            model_action = policy_runner.act(model_obs)
            check_shape("model_action", model_action, expected_last_dim=args.expected_action_dim)
            check_tensor_finite("model_action", model_action)
            check_device("model_action", model_action, args.model_device)

            env_action = act_adapter(model_action)
            check_shape("env_action", env_action, expected_last_dim=args.expected_action_dim)
            check_tensor_finite("env_action", env_action)
            check_device("env_action", env_action, args.device)

            if step == 0 or args.debug:
                print(f"[play_isaaclab] model observation shape: {tuple(model_obs.shape)}")
                print(f"[play_isaaclab] model action shape: {tuple(model_action.shape)}")
                print(f"[play_isaaclab] env action shape: {tuple(env_action.shape)}")
                if args.debug:
                    print_debug_tensor("model_obs", model_obs)
                    print_debug_tensor("model_action", model_action)
                    print_debug_tensor("env_action", env_action)

            raw_step = env.step(env_action)
            raw_obs, reward, done, info = _unwrap_step(raw_step)

            if args.render and hasattr(env, "render"):
                env.render()

            recorder.log_step(
                step=step,
                obs=model_obs,
                model_action=model_action,
                env_action=env_action,
                reward=reward,
                done=done,
                info=info,
            )

            done_any = False
            if done is not None:
                if isinstance(done, torch.Tensor):
                    done_any = bool(done.any().item())
                elif isinstance(done, (list, tuple)):
                    done_any = any(bool(x) for x in done)
                else:
                    done_any = bool(done)

            if done_any:
                raw_obs, _ = _unwrap_reset(env.reset())

    finally:
        recorder.save()
        if env is not None:
            env.close()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
