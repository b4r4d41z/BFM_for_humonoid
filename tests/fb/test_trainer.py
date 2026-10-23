import json
import subprocess
import sys

import torch

from bfm.fb.checkpoint import load_payload
from bfm.data.dataset import BFMDataset


def test_cli_training_and_exact_resumption(manifest, tmp_path):
    cfg = {"dataset_manifest": str(manifest), "output_dir": str(tmp_path / "split"),
           "seed": 31, "batch_size": 8, "training_steps": 4,
           "logging_interval": 1, "checkpoint_interval": 2, "reference_count": 12,
           "model": {"hidden_dim": 32, "latent_dim": 4}}
    path = tmp_path / "config.yaml"
    # JSON is a YAML subset; testing the actual YAML entry point without custom CLI shortcuts.
    path.write_text(json.dumps(cfg))
    command = [sys.executable, "scripts/train_offline_fb.py", "--config", str(path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    split_checkpoint = tmp_path / "split" / "last.pt"
    subprocess.run(command + ["--resume", str(split_checkpoint), "--training-steps", "8"], check=True, capture_output=True, text=True)
    subprocess.run(command + ["--output-dir", str(tmp_path / "full"), "--training-steps", "8"], check=True, capture_output=True, text=True)
    split, full = load_payload(split_checkpoint), load_payload(tmp_path / "full" / "last.pt")
    assert split["step"] == full["step"] == 8
    assert all(torch.equal(v, full["model"][k]) for k, v in split["model"].items())
    data = BFMDataset(manifest)
    train_states = torch.cat([x[key] for x in data.episodes for key in ("observation", "next_observation")])
    assert torch.allclose(split["model"]["normalizer.mean"], train_states.double().mean(0).float())
    assert split["training_episode_ids"] == ["toy0", "toy1"]
    metrics = [json.loads(line) for line in (tmp_path / "split" / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == list(range(1, 9))


def test_entrypoint_help_without_isaac():
    for script in ("scripts/train_offline_fb.py", "scripts/collect_bfm_zarr.py", "scripts/eval_zero_shot_fb.py"):
        result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
