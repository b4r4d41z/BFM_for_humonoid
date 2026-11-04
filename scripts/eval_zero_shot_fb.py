import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Autonomous measured Isaac rollouts with a frozen Vanilla FB policy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="Template-Test-Training-Direct-v0")
    parser.add_argument("--reward-task", choices=["reach_red_ball", "move_red_ball_to_container"], default="reach_red_ball")
    parser.add_argument("--goal-json", help="Optional complete measured goal state specification")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-control-steps", type=int, default=600)
    parser.add_argument("--reach-distance", type=float, default=.06)
    parser.add_argument("--target-half-extent", nargs=3, type=float, default=[.06, .06, .04])
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", default="runs/fb/evaluation.json")
    args = parser.parse_args()
    from isaaclab_wrapper.bfm_policy_runner import BFMPolicyRunner
    from isaaclab_wrapper.bfm_state import IsaacTransitionBridge
    from isaaclab_wrapper.env_factory import create_isaaclab_env
    from bfm.evaluation import evaluate_episode
    runner = BFMPolicyRunner(args.checkpoint, args.device)
    goal = json.loads(Path(args.goal_json).read_text()) if args.goal_json else {"type": "reward", "task": args.reward_task}
    runner.set_goal(goal)
    env, app = create_isaaclab_env(args.task, num_envs=1, device=args.device, headless=not args.render, render=args.render)
    bridge = None
    results = []
    try:
        bridge = IsaacTransitionBridge(env, runner.contract)
        for seed in args.seeds:
            results.append(evaluate_episode(env, bridge, runner, seed=seed, task=args.reward_task,
                                            max_steps=args.max_control_steps, reach_distance=args.reach_distance,
                                            target_half_extent=args.target_half_extent, hold_steps=args.hold_steps))
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "backend": "isaaclab", "checkpoint": args.checkpoint,
                "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                "training_step": runner.checkpoint["step"], "dataset_fingerprint": runner.checkpoint["dataset_fingerprint"],
                "training_episode_ids": runner.checkpoint["training_episode_ids"], "reference_indices": runner.checkpoint["reference_indices"],
                "goal": goal, "latent": runner.z.cpu().tolist(), "settings": vars(args), "episodes": results,
            }, indent=2, allow_nan=False))
    finally:
        if bridge is not None: bridge.close()
        env.close()
        app.close()


if __name__ == "__main__":
    main()
