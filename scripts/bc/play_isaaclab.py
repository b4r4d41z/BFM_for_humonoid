from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
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
from bc.data import schema as data_schema


def _extract_checkpoint_joint_names(meta: dict[str, Any]) -> list[str]:
    candidate_keys = ("joint_names", "meta_joint_names", "action_joint_names", "obs_joint_names")
    for key in candidate_keys:
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [str(x) for x in value]
    return []


def _load_joint_names_from_mapping_file(mapping_path: str) -> list[str]:
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML mapping requested but PyYAML is not installed. Use JSON file or install pyyaml."
            ) from exc
        payload = yaml.safe_load(text)

    if isinstance(payload, dict):
        for key in ("joint_names", "model_action_joint_names", "arm_joint_names"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                return [str(x) for x in value]
    raise ValueError(f"Could not find joint names list in mapping file: {mapping_path}")


def _extract_env_joint_names(env: Any) -> list[str]:
    names: list[str] = []
    if hasattr(env, "unwrapped"):
        env = env.unwrapped
    cfg = getattr(env, "cfg", None)
    if cfg is not None:
        cfg_names = getattr(cfg, "ctrl_joint_names", None)
        if isinstance(cfg_names, (list, tuple)):
            names = [str(x) for x in cfg_names]
    return names


def _compare_joint_orders(model_joint_names: list[str], env_joint_names: list[str]) -> str:
    if not model_joint_names:
        return "missing"
    if model_joint_names == env_joint_names:
        return "exact_match"
    if set(model_joint_names) & set(env_joint_names):
        return "partial"
    return "missing"


def _normalize_joint_names(names: list[str], mode: str) -> list[str]:
    if mode == "strict":
        return names
    if mode == "strip_lower":
        return [str(x).strip().lower() for x in names]
    raise ValueError(f"Unknown joint name normalization mode: {mode}")


def _write_contract_report(report: dict[str, Any], output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"contract_report_{ts}.json"
    md_path = out_dir / f"contract_report_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# IsaacLab BC Contract Report",
        "",
        f"- generated_utc: {report['generated_utc']}",
        f"- task: {report['task']}",
        f"- checkpoint: {report['checkpoint']}",
        "",
        "## Project 26D data/model contract",
        f"- compatible: {report['project_contract']['compatible']}",
        f"- state_layout: {report['project_contract']['state_layout']}",
        f"- action_layout: {report['project_contract']['action_layout']}",
        f"- action_type: {report['project_contract']['action_type']}",
        f"- gripper_mode: {report['project_contract']['gripper_mode']}",
        f"- hand_open_prototype_6: {report['project_contract']['hand_open_prototype_6']}",
        f"- hand_closed_prototype_6: {report['project_contract']['hand_closed_prototype_6']}",
        f"- left_hand_slice: {report['project_contract']['left_hand_slice']}",
        f"- right_hand_slice: {report['project_contract']['right_hand_slice']}",
        "",
        "## Dimensions",
        f"- model_obs_dim: {report['dims']['model_obs_dim']}",
        f"- model_action_dim: {report['dims']['model_action_dim']}",
        f"- env_observation_space: {report['dims']['env_observation_space']}",
        f"- env_action_space: {report['dims']['env_action_space']}",
        "",
        "## Env controlled joints",
    ]
    lines.extend([f"- {name}" for name in report["env_ctrl_joint_names"]])
    lines.extend(
        [
            "",
            "## Action channel order source",
            f"- dataset_schema_arm_hand_split: {report['action_channel_order']['dataset_schema_split']}",
            f"- checkpoint_joint_names_status: {report['action_channel_order']['checkpoint_joint_names_status']}",
            f"- mapping_status: {report['action_channel_order']['mapping_status']}",
            f"- notes: {report['action_channel_order']['notes']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[play_isaaclab] contract report saved: {json_path}")
    print(f"[play_isaaclab] contract report saved: {md_path}")


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
    parser.add_argument("--auto_adjust_obs_dim", action="store_true", default=True)
    parser.add_argument("--no_auto_adjust_obs_dim", action="store_false", dest="auto_adjust_obs_dim")

    parser.add_argument("--obs_dim", type=int, default=None, help="Fallback model construction obs dim")
    parser.add_argument("--action_dim", type=int, default=None, help="Fallback model construction action dim")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Fallback MLP hidden dim")
    parser.add_argument("--hidden_layers", type=int, default=2, help="Fallback MLP hidden layers")

    parser.add_argument("--action_scale", type=float, default=1.0)
    parser.add_argument("--clip_actions", action="store_true", default=True)
    parser.add_argument("--no_clip_actions", action="store_false", dest="clip_actions")

    parser.add_argument("--record_rollout", action="store_true")
    parser.add_argument("--rollout_output_dir", type=str, default="runs/bc/isaaclab_rollouts")
    parser.add_argument("--contract_report_dir", type=str, default="runs/bc/isaaclab_contract")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--action_mode", type=str, default="arm_only", choices=("arm_only", "arm_plus_gripper_bridge", "identity"))
    parser.add_argument("--action_joint_source", type=str, default="checkpoint_meta", choices=("checkpoint_meta", "mapping_file"))
    parser.add_argument("--action_mapping_file", type=str, default=None, help="Path to JSON/YAML with model action joint names")
    parser.add_argument("--allow_provisional_mapping", action="store_true", help="Allow fallback [0:14] mapping when exact name match is unavailable")
    parser.add_argument("--joint_name_normalization", type=str, default="strict", choices=("strict", "strip_lower"))
    parser.add_argument("--gripper_bridge_aggregator", type=str, default="prototype_distance", choices=("prototype_distance",))
    parser.add_argument("--gripper_open_threshold", type=float, default=0.35, help="closure score threshold to open (<=)")
    parser.add_argument("--gripper_close_threshold", type=float, default=0.65, help="closure score threshold to close (>=)")
    parser.add_argument("--gripper_open_prototype", type=float, nargs=6, default=[float(x) for x in data_schema.HAND_OPEN_PROTOTYPE_6])
    parser.add_argument("--gripper_closed_prototype", type=float, nargs=6, default=[float(x) for x in data_schema.HAND_CLOSED_PROTOTYPE_6])
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
            auto_adjust_dim=args.auto_adjust_obs_dim,
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

        expected_obs_dim = args.expected_obs_dim
        expected_action_dim = args.expected_action_dim
        if expected_obs_dim is None and policy_runner.expected_obs_dim:
            expected_obs_dim = policy_runner.expected_obs_dim
        if expected_action_dim is None and policy_runner.expected_action_dim:
            expected_action_dim = policy_runner.expected_action_dim

        env_cfg = getattr(env.unwrapped if hasattr(env, "unwrapped") else env, "cfg", None)
        env_action_dim = int(getattr(env_cfg, "action_space", 0)) if env_cfg is not None else None
        env_ctrl_joint_names = _extract_env_joint_names(env)
        if args.action_joint_source == "mapping_file":
            if not args.action_mapping_file:
                raise ValueError("--action_mapping_file is required when --action_joint_source=mapping_file")
            checkpoint_joint_names = _load_joint_names_from_mapping_file(args.action_mapping_file)
        else:
            checkpoint_joint_names = _extract_checkpoint_joint_names(policy_runner.checkpoint_meta)

        normalized_model_joint_names = _normalize_joint_names(checkpoint_joint_names, args.joint_name_normalization)
        normalized_env_joint_names = _normalize_joint_names(env_ctrl_joint_names, args.joint_name_normalization)

        # Rebuild adapters with inferred dims for stricter checks and better key selection.
        obs_adapter = ObservationAdapter(
            expected_obs_dim=expected_obs_dim,
            device=args.model_device,
            debug=args.debug,
            auto_adjust_dim=args.auto_adjust_obs_dim,
        )
        act_adapter = ActionAdapter(
            expected_action_dim=expected_action_dim,
            env_device=args.device,
            action_scale=args.action_scale,
            clip_actions=args.clip_actions,
            debug=args.debug,
            action_mode=args.action_mode,
            model_action_dim=expected_action_dim,
            env_action_dim=env_action_dim,
            model_action_joint_names=normalized_model_joint_names,
            env_ctrl_joint_names=normalized_env_joint_names,
            allow_schema_fallback=args.allow_provisional_mapping,
            gripper_bridge_aggregator=args.gripper_bridge_aggregator,
            gripper_open_threshold=args.gripper_open_threshold,
            gripper_close_threshold=args.gripper_close_threshold,
            gripper_open_prototype=args.gripper_open_prototype,
            gripper_closed_prototype=args.gripper_closed_prototype,
        )
        print(f"[play_isaaclab] expected obs dim: {expected_obs_dim}")
        print(f"[play_isaaclab] expected action dim: {expected_action_dim}")

        unwrapped_env = env.unwrapped if hasattr(env, "unwrapped") else env
        env_cfg = getattr(unwrapped_env, "cfg", None)
        env_action_space = getattr(env_cfg, "action_space", None)
        env_observation_space = getattr(env_cfg, "observation_space", None)
        mapping_status = _compare_joint_orders(normalized_model_joint_names, normalized_env_joint_names)
        verification_status = "verified" if mapping_status == "exact_match" else "provisional"
        notes = (
            "Dataset schema defines split arm=[0:14], hand=[14:26]. "
            "If checkpoint joint names are absent or mismatched, arm mapping remains provisional."
        )

        print("[play_isaaclab] ===== CONTRACT: model dims =====")
        print(f"[play_isaaclab] model expected_obs_dim={expected_obs_dim}, expected_action_dim={expected_action_dim}")
        print("[play_isaaclab] ===== CONTRACT: env dims =====")
        print(f"[play_isaaclab] env observation_space={env_observation_space}, env action_space={env_action_space}")
        print("[play_isaaclab] ===== CONTRACT: env joint order =====")
        print(f"[play_isaaclab] env ctrl_joint_names({len(env_ctrl_joint_names)}): {env_ctrl_joint_names}")
        print("[play_isaaclab] ===== CONTRACT: model action channel order source =====")
        print(
            "[play_isaaclab] dataset schema split: arm=[0:14], hand=[14:26], "
            f"checkpoint_joint_names_status={mapping_status}, verification={verification_status}"
        )
        if verification_status != "verified":
            print(f"[play_isaaclab][WARNING] action mapping is provisional. {notes}")
            if not args.allow_provisional_mapping:
                raise RuntimeError(
                    "Mapping is not exact_match and provisional mapping is disabled. "
                    "Provide a verified joint mapping source or pass --allow_provisional_mapping."
                )

        project_contract = data_schema.validate_contract_metadata(
            data_schema.get_default_contract_metadata(), strict=True
        )
        contract_compatible = (
            expected_obs_dim in (None, project_contract["obs_dim"])
            and expected_action_dim in (None, project_contract["action_dim"])
        )

        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "task": args.task,
            "checkpoint": args.checkpoint,
            "project_contract": {
                "compatible": contract_compatible,
                "obs_dim": project_contract["obs_dim"],
                "action_dim": project_contract["action_dim"],
                "arm_dim": project_contract["arm_dim"],
                "hand_dim": project_contract["hand_dim"],
                "arm_joint_names": project_contract["arm_joint_names"],
                "hand_value_names": project_contract["hand_value_names"],
                "state_layout": project_contract["state_layout"],
                "action_layout": project_contract["action_layout"],
                "action_type": project_contract["action_type"],
                "gripper_mode": project_contract["gripper_mode"],
                "hand_open_prototype_6": project_contract["hand_open_prototype_6"],
                "hand_closed_prototype_6": project_contract["hand_closed_prototype_6"],
                "left_hand_slice": project_contract["left_hand_slice"],
                "right_hand_slice": project_contract["right_hand_slice"],
                "image_keys": project_contract["image_keys"],
            },
            "dims": {
                "model_obs_dim": expected_obs_dim,
                "model_action_dim": expected_action_dim,
                "env_observation_space": env_observation_space,
                "env_action_space": env_action_space,
            },
            "env_ctrl_joint_names": env_ctrl_joint_names,
            "action_channel_order": {
                "dataset_schema_split": {
                    "arm": [0, data_schema.ACTION_ARM_DIM],
                    "hand": [data_schema.ACTION_ARM_DIM, data_schema.ACTION_FULL_DIM],
                },
                "checkpoint_joint_names_status": mapping_status,
                "mapping_status": verification_status,
                "checkpoint_joint_names": checkpoint_joint_names,
                "action_joint_source": args.action_joint_source,
                "action_mapping_file": args.action_mapping_file,
                "joint_name_normalization": args.joint_name_normalization,
                "notes": notes,
            },
        }
        _write_contract_report(report=report, output_dir=args.contract_report_dir)

        raw_obs, _ = _unwrap_reset(env.reset())
        print(f"[play_isaaclab] raw observation type: {type(raw_obs).__name__}")

        for step in range(int(args.max_steps)):
            try:
                model_obs = obs_adapter(raw_obs)
                check_shape("model_obs", model_obs, expected_last_dim=expected_obs_dim)
                check_tensor_finite("model_obs", model_obs)
                check_device("model_obs", model_obs, args.model_device)

                model_action = policy_runner.act(model_obs)
                check_shape("model_action", model_action, expected_last_dim=expected_action_dim)
                check_tensor_finite("model_action", model_action)
                check_device("model_action", model_action, args.model_device)

                env_action = act_adapter(model_action)
                check_shape("env_action", env_action, expected_last_dim=env_action_dim)
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
                        if args.action_mode == "arm_plus_gripper_bridge":
                            print(f"[play_isaaclab][gripper_bridge] {act_adapter.gripper_bridge_stats}")

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
            except Exception as exc:
                print(f"[play_isaaclab] ERROR at step={step}: {exc}")
                traceback.print_exc()
                raise

    finally:
        recorder.save()
        if env is not None:
            env.close()
        if sim_app is not None:
            sim_app.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
