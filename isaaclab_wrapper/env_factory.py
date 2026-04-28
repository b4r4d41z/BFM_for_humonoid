from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def add_test_training_extension_to_syspath() -> Path:
    """Allow importing `Test_training.tasks` without editable install."""
    repo_root = Path(__file__).resolve().parent.parent
    extension_root = repo_root / "Test_training" / "source" / "Test_training"
    if extension_root.exists() and str(extension_root) not in sys.path:
        sys.path.insert(0, str(extension_root))
    return extension_root


def create_isaaclab_env(
    task: str,
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = True,
    render: bool = False,
) -> tuple[Any, Any]:
    """Launch Isaac Lab app and create environment from task name."""
    try:
        from isaaclab.app import AppLauncher  # type: ignore
    except Exception:
        from omni.isaac.lab.app import AppLauncher  # type: ignore

    # Ensure test task registration package can be imported.
    add_test_training_extension_to_syspath()

    app_launcher = AppLauncher(headless=bool(headless and not render), device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym

    try:
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg  # type: ignore
    except Exception:
        import omni.isaac.lab_tasks as isaaclab_tasks  # type: ignore  # noqa: F401
        from omni.isaac.lab_tasks.utils import parse_env_cfg  # type: ignore

    import Test_training.tasks  # noqa: F401

    env_cfg = parse_env_cfg(task, num_envs=num_envs, device=device)
    env = gym.make(task, cfg=env_cfg)
    return env, simulation_app
