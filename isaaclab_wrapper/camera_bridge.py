from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


_CAMERA_ROLES = ("head", "left_wrist", "right_wrist")
_ENV_RE = re.compile(r"(/envs/env_)(\d+)(/)")


class IsaacCameraBridge:
    """Discover existing IsaacSim USD cameras and read RGB tensors for policy inference."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str,
        image_width: int = 224,
        image_height: int = 224,
        head_camera_prim: str | None = None,
        left_wrist_camera_prim: str | None = None,
        right_wrist_camera_prim: str | None = None,
        debug: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.debug = bool(debug)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(f"Camera image size must be positive, got {self.image_width}x{self.image_height}")

        self._stage = self._get_stage()
        self.discovered_camera_prims = self.discover_camera_prims(self._stage)
        self._print_discovered_cameras()

        overrides = {
            "head": head_camera_prim,
            "left_wrist": left_wrist_camera_prim,
            "right_wrist": right_wrist_camera_prim,
        }
        self.camera_prim_paths = self._resolve_camera_paths(overrides)
        self._print_resolved_cameras()

        self._annotators: dict[str, list[Any]] = {role: [] for role in _CAMERA_ROLES}
        self._render_products: dict[str, list[Any]] = {role: [] for role in _CAMERA_ROLES}
        self._setup_render_products()

    @staticmethod
    def _get_stage() -> Any:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Could not access the active USD stage for camera discovery")
        return stage

    @staticmethod
    def _is_camera_prim(prim: Any) -> bool:
        if not prim.IsValid():
            return False
        if prim.GetTypeName() == "Camera":
            return True
        from pxr import UsdGeom

        return bool(prim.IsA(UsdGeom.Camera))

    @classmethod
    def discover_camera_prims(cls, stage: Any | None = None) -> list[str]:
        if stage is None:
            stage = cls._get_stage()
        camera_paths: list[str] = []
        for prim in stage.Traverse():
            if cls._is_camera_prim(prim):
                camera_paths.append(str(prim.GetPath()))
        return sorted(camera_paths)

    def _print_discovered_cameras(self) -> None:
        print(f"[IsaacCameraBridge] discovered_camera_prims({len(self.discovered_camera_prims)}):")
        for path in self.discovered_camera_prims:
            print(f"[IsaacCameraBridge]   {path}")
        if not self.discovered_camera_prims:
            raise RuntimeError(
                "No USD Camera prims were found after IsaacLab scene creation. "
                "Verify the robot USD contains Camera prims or pass explicit camera prim paths."
            )

    def _print_resolved_cameras(self) -> None:
        print("[IsaacCameraBridge] resolved policy camera prims:")
        for role in _CAMERA_ROLES:
            print(f"[IsaacCameraBridge]   {role}: {self.camera_prim_paths[role]}")

    def _validate_camera_path(self, path: str, role: str) -> None:
        prim = self._stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                f"Configured {role} camera prim does not exist: {path}. "
                f"Discovered camera prims: {self.discovered_camera_prims}"
            )
        if not self._is_camera_prim(prim):
            raise RuntimeError(f"Configured {role} camera prim is not a USD Camera: {path}")

    def _expand_env_camera_paths(self, path: str, role: str) -> list[str]:
        match = _ENV_RE.search(path)
        if self.num_envs == 1 or match is None:
            self._validate_camera_path(path, role)
            return [path for _ in range(self.num_envs)]

        expanded: list[str] = []
        for env_idx in range(self.num_envs):
            env_path = _ENV_RE.sub(rf"\g<1>{env_idx}\g<3>", path, count=1)
            self._validate_camera_path(env_path, role)
            expanded.append(env_path)
        return expanded

    @staticmethod
    def _env_pattern(path: str) -> str:
        return _ENV_RE.sub(r"\g<1>*\g<3>", path, count=1)

    @staticmethod
    def _role_score(path: str, role: str) -> int:
        text = path.lower()
        tokens = re.split(r"[^a-z0-9]+", text)
        token_set = set(tokens)
        is_left = any(key in text for key in ("left", "zarm_l", "_l_", "/l_", "-l-")) or "l" in token_set
        is_right = any(key in text for key in ("right", "zarm_r", "_r_", "/r_", "-r-")) or "r" in token_set
        is_wrist_or_hand = any(key in text for key in ("wrist", "hand", "finger", "gripper", "zarm"))
        has_camera = "camera" in text or "cam" in token_set
        score = 0
        if role == "head":
            if "head" in text:
                score += 8
            if has_camera:
                score += 2
            if is_left or is_right or is_wrist_or_hand:
                score -= 6
            if "camera" in token_set:
                score += 1
        elif role == "left_wrist":
            if is_left:
                score += 6
            if is_wrist_or_hand:
                score += 4
            if has_camera:
                score += 2
            if is_right:
                score -= 8
        elif role == "right_wrist":
            if is_right:
                score += 6
            if is_wrist_or_hand:
                score += 4
            if has_camera:
                score += 2
            if is_left:
                score -= 8
        return score

    def _auto_resolve_role(self, role: str) -> str:
        scored = [(self._role_score(path, role), path) for path in self.discovered_camera_prims]
        candidates = [(score, path) for score, path in scored if score > 0]
        if not candidates:
            raise RuntimeError(
                f"Could not automatically resolve {role} camera. "
                f"Discovered camera prims: {self.discovered_camera_prims}. "
                f"Pass --{role}_camera_prim with the correct path."
            )

        best_score = max(score for score, _ in candidates)
        best_paths = sorted(path for score, path in candidates if score == best_score)
        best_patterns = sorted({self._env_pattern(path) for path in best_paths})
        if len(best_patterns) != 1:
            role_candidates = [path for _, path in sorted(candidates, key=lambda item: (-item[0], item[1]))]
            raise RuntimeError(
                f"Ambiguous {role} camera discovery. Best candidate patterns={best_patterns}. "
                f"All {role} candidates={role_candidates}. "
                f"All discovered camera prims={self.discovered_camera_prims}. "
                f"Pass --{role}_camera_prim with the correct path."
            )
        return best_paths[0]

    def _resolve_camera_paths(self, overrides: dict[str, str | None]) -> dict[str, list[str]]:
        resolved: dict[str, list[str]] = {}
        for role in _CAMERA_ROLES:
            selected = overrides.get(role) or self._auto_resolve_role(role)
            resolved[role] = self._expand_env_camera_paths(str(selected), role)
        return resolved

    def _setup_render_products(self) -> None:
        import omni.replicator.core as rep

        for role in _CAMERA_ROLES:
            for camera_path in self.camera_prim_paths[role]:
                render_product = rep.create.render_product(
                    camera_path,
                    resolution=(self.image_width, self.image_height),
                )
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach([render_product])
                self._render_products[role].append(render_product)
                self._annotators[role].append(annotator)

    @staticmethod
    def _rgb_data_to_hwc(data: Any, role: str, camera_path: str) -> np.ndarray:
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        arr = np.asarray(data)
        if arr.size == 0:
            raise RuntimeError(f"RGB annotator returned an empty frame for {role} camera {camera_path}")
        if arr.ndim != 3:
            raise RuntimeError(
                f"RGB frame for {role} camera {camera_path} must be [H,W,C] or [C,H,W], got shape {arr.shape}"
            )
        if arr.shape[-1] in (3, 4):
            hwc = arr[..., :3]
        elif arr.shape[0] in (3, 4):
            hwc = np.moveaxis(arr[:3, ...], 0, -1)
        else:
            raise RuntimeError(
                f"RGB frame for {role} camera {camera_path} must have 3/4 channels, got shape {arr.shape}"
            )
        return np.ascontiguousarray(hwc)

    def get_images(self) -> dict[str, torch.Tensor]:
        images: dict[str, torch.Tensor] = {}
        for role in _CAMERA_ROLES:
            frames = []
            for env_idx, annotator in enumerate(self._annotators[role]):
                camera_path = self.camera_prim_paths[role][env_idx]
                frame = self._rgb_data_to_hwc(annotator.get_data(), role, camera_path)
                frames.append(frame)
            batch = np.stack(frames, axis=0)
            images[role] = torch.as_tensor(batch, device=self.device)
            if images[role].ndim != 4 or images[role].shape[-1] != 3:
                raise RuntimeError(f"obs.images.{role} must be [B,H,W,3], got {tuple(images[role].shape)}")
        return images

    @staticmethod
    def _to_uint8_image(frame: torch.Tensor) -> np.ndarray:
        arr = frame.detach().cpu().numpy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise RuntimeError(f"Debug camera frame must be [H,W,3], got shape {arr.shape}")
        return arr

    def save_debug_frames(self, images: dict[str, torch.Tensor], output_dir: str | Path) -> None:
        missing = [role for role in _CAMERA_ROLES if role not in images]
        if missing:
            raise RuntimeError(f"Cannot save debug camera frames; missing image keys: {missing}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if importlib.util.find_spec("PIL") is not None:
            from PIL import Image

            for role in _CAMERA_ROLES:
                frame = self._to_uint8_image(images[role][0])
                Image.fromarray(frame, mode="RGB").save(output_path / f"{role}.png")
        elif importlib.util.find_spec("imageio") is not None:
            import imageio.v2 as imageio

            for role in _CAMERA_ROLES:
                frame = self._to_uint8_image(images[role][0])
                imageio.imwrite(output_path / f"{role}.png", frame)
        else:
            raise RuntimeError("Saving debug camera frames requires Pillow or imageio to be installed")

        print(f"[IsaacCameraBridge] saved debug camera frames to {output_path}")
