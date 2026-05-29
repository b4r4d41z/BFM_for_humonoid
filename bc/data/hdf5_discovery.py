from __future__ import annotations

from pathlib import Path


def _is_h5_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".h5"


def _directory_h5_files(path: Path, *, recursive: bool) -> list[Path]:
    iterator = path.rglob("*") if recursive else path.iterdir()
    return [candidate for candidate in iterator if _is_h5_file(candidate)]


def discover_h5_files(paths: list[Path], *, recursive: bool = False) -> list[Path]:
    """Discover unique, sorted HDF5 files from files and directories.

    ``paths`` may include individual ``.h5`` files, directories that contain
    ``.h5`` files directly, and (when ``recursive`` is true) directories with
    nested subdirectories containing ``.h5`` files. Paths are handled as
    ``pathlib.Path`` objects so GVFS/SMB paths containing ``:`` or ``,`` are
    not split or parsed manually.
    """
    files: list[Path] = []

    for path in paths:
        if path.is_file():
            if path.suffix.lower() != ".h5":
                raise ValueError(f"Expected .h5 file, got: {path}")
            files.append(path)
            continue

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not path.is_dir():
            raise ValueError(f"Expected .h5 file or directory, got: {path}")

        files.extend(_directory_h5_files(path, recursive=recursive))

    uniq_sorted = sorted(set(files), key=lambda p: str(p))
    if len(uniq_sorted) == 0:
        searched = "\n".join(f"  - {path}" for path in paths)
        recursive_state = "enabled" if recursive else "disabled"
        suggestion = ""
        if not recursive:
            suggestion = "\nSuggestion: use --recursive if .h5 files are inside nested subdirectories."
        raise FileNotFoundError(
            "No .h5 files found.\n"
            f"Searched paths:\n{searched}\n"
            f"Recursive mode: {recursive_state}."
            f"{suggestion}"
        )

    return uniq_sorted


def limit_h5_files(files: list[Path], max_files: int | None = None) -> list[Path]:
    if max_files is None:
        return files
    if max_files <= 0:
        raise ValueError(f"max_files must be positive, got: {max_files}")
    return files[:max_files]


def collect_h5_files(paths: list[Path], max_files: int | None = None, recursive: bool = False) -> list[Path]:
    return limit_h5_files(discover_h5_files(paths, recursive=recursive), max_files=max_files)


def print_h5_dataset_summary(
    *,
    data_roots: list[Path],
    recursive: bool,
    discovered_h5_files: list[Path],
    selected_h5_files: list[Path] | None = None,
    max_files: int | None = None,
    preview_count: int = 10,
    prefix: str = "[BFM offline train]",
) -> None:
    selected = selected_h5_files if selected_h5_files is not None else discovered_h5_files

    print(f"{prefix} dataset discovery summary:")
    print(f"{prefix} data roots:")
    for path in data_roots:
        print(f"{prefix}   - {path}")
    print(f"{prefix} recursive: {recursive}")
    print(f"{prefix} total discovered .h5 files: {len(discovered_h5_files)}")
    if max_files is not None:
        print(f"{prefix} max_files: {max_files}")
    print(f"{prefix} selected .h5 files for this run: {len(selected)}")
    print(f"{prefix} first {min(preview_count, len(discovered_h5_files))} discovered files:")
    for i, path in enumerate(discovered_h5_files[:preview_count]):
        print(f"{prefix}   [{i}] {path}")
