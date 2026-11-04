"""Direct measured Isaac export. Existing command-proxy files are never relabelled."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="Record measured KUAVO policy-rate transitions into BFM schema v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="Template-Test-Training-Direct-v0")
    parser.add_argument("--eef-body", required=True, help="Exact articulation body name, validated at runtime")
    parser.add_argument("--eef-offset", type=float, nargs=3, default=[0., 0., 0.], help="TCP offset in body-local metres; default is body origin")
    parser.add_argument("--left-hold", type=float, nargs=7)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--actions-npz", help="actions array [policy_steps,8] in [-1,1]; commands will actually be executed in Isaac")
    source.add_argument("--random-smoke", action="store_true", help="Small joint-target perturbations for recorder validation, not a manipulation demonstration")
    parser.add_argument("--max-policy-steps", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--policy-hz", type=float, default=10.)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import numpy as np
    import torch
    from isaaclab_wrapper.env_factory import create_isaaclab_env
    from isaaclab_wrapper.bfm_state import contract_from_env, IsaacTransitionBridge
    from bfm.contracts.kuavo import BFMActionAdapter
    from bfm.data.recorder import BFMRecorder, cpu_state
    from bfm.data.schema import SCHEMA_VERSION
    actions = None
    if args.actions_npz:
        with np.load(args.actions_npz, allow_pickle=False) as archive:
            actions = archive["actions"]
        if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions) or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
            raise ValueError("Action file must provide finite [N,8] in [-1,1]")
    if args.max_policy_steps < 1:
        raise ValueError("max-policy-steps must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": SCHEMA_VERSION, "dataset_id": str(uuid.uuid4()), "episodes": []}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Existing manifest has incompatible schema")
    env, app = create_isaaclab_env(args.task, num_envs=1, device=args.device)
    bridge = None
    try:
        contract = contract_from_env(env.unwrapped, eef_body=args.eef_body, eef_offset=args.eef_offset,
                                     policy_hz=args.policy_hz, left_hold=args.left_hold)
        bridge = IsaacTransitionBridge(env, contract)
        adapter = BFMActionAdapter(contract, env.unwrapped._ctrl_joint_names, args.device)
        stride = round(contract.policy_dt / contract.control_dt)
        for seed in args.seeds:
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            state = bridge.extract()
            episode_id = f"seed{seed}-{uuid.uuid4().hex[:12]}"
            provenance = {"kind": "isaac_random_smoke" if args.random_smoke else "isaac_executed_commands", "seed": seed, "task": args.task}
            if args.actions_npz:
                provenance["command_file_sha256"] = hashlib.sha256(Path(args.actions_npz).read_bytes()).hexdigest()
            recorder = BFMRecorder(contract, episode_id=episode_id, split=args.split, provenance=provenance,
                                   initial_scene={k: v.tolist() for k, v in cpu_state(state).items()},
                                   joint_names=env.unwrapped.robot.joint_names)
            count = min(args.max_policy_steps, len(actions)) if actions is not None else args.max_policy_steps
            for step in range(count):
                if actions is not None:
                    normalized = torch.as_tensor(actions[step:step+1], device=args.device)
                else:
                    low, high = adapter.low, adapter.high
                    current = 2 * (state["right_arm_q"] - low) / (high-low) - 1
                    noise = torch.as_tensor(rng.uniform(-.1, .1, (1, 7)), device=args.device, dtype=torch.float32)
                    normalized = torch.cat(((current + noise).clamp(-1, 1), -torch.ones((1, 1), device=args.device)), -1)
                command = adapter(normalized)
                applied = None
                for substep in range(stride):
                    next_state, term, trunc, executed, _ = bridge.step(command)
                    if applied is not None and not torch.equal(applied[1], executed[1]):
                        raise ValueError("Executed action changed during a held policy interval")
                    applied = executed
                    if bool(term[0] or trunc[0]): break
                terminal, timeout = bool(term[0]), bool(trunc[0])
                cutoff = step == count - 1 and not terminal
                recorder.append(state, *applied, next_state, terminated=terminal,
                                truncated=timeout or cutoff, control_steps=substep + 1)
                state = next_state
                if terminal or timeout: break
            name = episode_id + ".zarr"
            print(recorder.save(output / name))
            manifest["episodes"].append({"path": name, "split": args.split})
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(manifest_path)
    finally:
        if bridge is not None: bridge.close()
        env.close()
        app.close()


if __name__ == "__main__":
    main()
