"""Fail-closed orchestration of the existing Sharpa converter and dataset."""
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import yaml
import zarr

from .mcap import convert
from .sharpa import SharpaDataset, validate


def discover(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f'Not a recordings directory: {directory}')
    episodes = sorted({p.parent for p in directory.rglob('episode.json')})
    if (directory/'episode.json').is_file():
        episodes = sorted(set(episodes) | {directory})
    if not episodes:
        raise ValueError(f'No episode.json recordings found in {directory}')
    return episodes


def source_identity(episode, policy):
    metadata = json.loads((episode/'episode.json').read_text())
    code = metadata.get('recording_code', {})
    commit = code.get('commit_sha', '')
    if code.get('status') != 'available' or len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Authoritative recording_code.commit_sha is unavailable/invalid')
    identity = metadata['bfm']['episode']['episode_id']
    if not isinstance(identity, str) or not identity:
        raise ValueError('Missing episode identity')
    bag = episode/'bag'
    bag_metadata = yaml.safe_load((bag/'metadata.yaml').read_text())
    hashes = {}
    for name in bag_metadata['rosbag2_bagfile_information']['relative_file_paths']:
        path = (bag/name).resolve()
        if not path.is_relative_to(bag.resolve()):
            raise ValueError('MCAP path escapes bag directory')
        with path.open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not hashes:
        raise ValueError('No MCAP files declared')
    # Reuse requires identical source bytes, policy and converter implementation.
    digest = hashlib.sha256(json.dumps([metadata, bag_metadata, hashes, policy], sort_keys=True).encode())
    for module in ('mcap.py', 'sharpa.py', 'policy_time.py'):
        digest.update(Path(__file__).with_name(module).read_bytes())
    from bfm.contracts import sharpa
    digest.update(Path(sharpa.__file__).read_bytes())
    return metadata, commit, 'episode-' + hashlib.sha256(identity.encode()).hexdigest() + '.zarr', digest.hexdigest()


def progress(index, total, name, state):
    fraction = index/total
    filled = int(20*fraction)
    print(f'{state:10} {index}/{total} {name} [{"#"*filled}{"-"*(20-filled)}] {fraction:.0%}', flush=True)


def prepare(directory, destination, policy, *, report=progress):
    episodes = discover(directory)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    for index, episode in enumerate(episodes, 1):
        report(index, len(episodes), episode.name, 'converting')
        try:
            metadata, commit, name, key = source_identity(episode, policy)
            identity = metadata['bfm']['episode']['episode_id']
            if identity in seen:
                raise ValueError('Duplicate raw episode identity')
            seen.add(identity)
            output = destination/name
            if output.exists():
                root = zarr.open_group(str(output), mode='r')
                validate(root)
                if root.attrs.get('conversion_key') != key or root.attrs['source_metadata'] != metadata:
                    raise ValueError(f'Existing output differs from raw source/policy/converter; move it aside explicitly: {output}')
                report(index, len(episodes), episode.name, 'skipped')
                continue
            # Converter already writes atomically. This outer transaction also
            # includes reuse provenance and validation before publishing the store.
            with tempfile.TemporaryDirectory(prefix='.conversion-', dir=destination) as scratch:
                staged = Path(scratch)/name
                with tempfile.TemporaryFile(mode='w+') as diagnostics:
                    try:
                        with redirect_stdout(diagnostics):
                            convert(episode, staged, collection_commit=commit, **policy)
                    except Exception:
                        diagnostics.seek(0)
                        detail = diagnostics.read()
                        if detail:
                            print(detail, end='')
                        raise
                root = zarr.open_group(str(staged), mode='a')
                root.attrs['conversion_key'] = key
                validate(root)
                if output.exists():
                    raise FileExistsError(output)
                os.rename(staged, output)
            report(index, len(episodes), episode.name, 'validated')
        except Exception as exc:
            report(index, len(episodes), episode.name, 'failed')
            raise ValueError(f'{episode}: {exc}') from exc
    return manifest_for(destination)


def manifest_for(destination):
    destination = Path(destination)
    manifest = destination/'manifest.json'
    existing = json.loads(manifest.read_text()) if manifest.exists() else {}
    assignments = {}
    for entry in existing.get('episodes', []):
        path = (destination/entry['path']).resolve()
        if path in assignments or entry['split'] not in ('train', 'validation', 'test') or not path.exists():
            raise ValueError('Invalid or missing existing split assignment')
        assignments[path] = entry['split']
    entries = []
    for path in sorted(destination.glob('*.zarr')):
        validate(zarr.open_group(str(path), mode='r'))
        entries.append({'path':path.name, 'split':assignments.get(path.resolve(), 'train')})
    if set(assignments) - {p.resolve() for p in destination.glob('*.zarr')}:
        raise ValueError('Existing manifest references stores outside canonical data/zarr')
    content = {'schema_version':'3.0.0', 'dataset_id':existing.get('dataset_id', 'recordings-sharpa-v3'), 'episodes':entries}
    temporary = destination/f'.manifest-{uuid.uuid4().hex}.json'
    try:
        temporary.write_text(json.dumps(content, indent=2)+'\n')
        SharpaDataset(temporary, observation_profile='vision_proprio_v1')
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest
