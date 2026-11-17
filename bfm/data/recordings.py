"""Fail-closed orchestration of the existing Sharpa converter and dataset."""
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np
import yaml
import zarr
from mcap.exceptions import McapError

from .mcap import convert
from .sharpa import SharpaDataset, validate


class BatchCompatibilityError(ValueError):
    """Existing outputs or requested dataset contracts cannot safely be combined."""


class NoSuccessfulRecordings(ValueError):
    """All inputs were processed, but no publishable dataset exists."""


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
    if not isinstance(code,dict):raise ValueError('Invalid recording_code metadata')
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
    for module in ('mcap.py', 'mcap_v4.py', 'execution_v4.py', 'decoupled.py', 'source_provenance.py', 'camera_sync.py', 'mcap_decoupled.py', 'sharpa.py', 'policy_time.py'):
        digest.update(Path(__file__).with_name(module).read_bytes())
    from bfm.contracts import sharpa
    digest.update(Path(sharpa.__file__).read_bytes())
    return metadata, commit, 'episode-' + hashlib.sha256(identity.encode()).hexdigest() + '.zarr', digest.hexdigest()


def progress(index, total, name, state, *, transitions=None, error=None):
    if state=='converting':return
    label={'validated':'CONVERTED','converted_with_skips':'CONVERTED_WITH_SKIPS','skipped':'REUSED','failed':'FAILED'}[state]
    detail=f' transitions={transitions}' if transitions is not None else f' {error}' if error else ''
    print(f'[{index:02d}/{total:02d}] {label:20} {name}{detail}',flush=True)


def _write_report(destination, result, report):
    records=result['recordings']
    result['summary']=dict(total_recordings=result['total_recordings'],processed=len(records),
        converted=sum(r['status'] in ('CONVERTED','CONVERTED_WITH_SKIPS') for r in records),
        converted_with_skips=sum(r['status']=='CONVERTED_WITH_SKIPS' for r in records),
        reused=sum(r['status']=='REUSED' for r in records),failed=sum(r['status']=='FAILED' for r in records),
        valid_transitions=sum(r['valid_transitions'] or 0 for r in records if r['status']!='FAILED'))
    temporary=destination/f'.report-{uuid.uuid4().hex}.json'
    try:
        temporary.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        os.replace(temporary,destination/'conversion_report.json')
    finally:temporary.unlink(missing_ok=True)
    if report is progress:
        s=result['summary']
        print('\nConversion '+('aborted' if result.get('aborted') else 'completed'))
        for label,key in [('Total recordings','total_recordings'),('Processed','processed'),('Converted','converted'),('Reused','reused'),('Failed','failed'),('Valid transitions','valid_transitions')]:
            print(f'{label}: {s[key]}')
        if s['failed']:
            print('\nFailed recordings:')
            for r in records:
                if r['status']=='FAILED':print(f"  {r['path']}: {r['failure']['message']}")
        print(f"Report: {destination/'conversion_report.json'}",flush=True)


def prepare(directory, destination, policy, *, report=progress, expected_version=None, expected_architecture=None):
    # Configuration and destination errors are batch-wide, not bad recordings.
    if not isinstance(policy,dict) or set(policy)-{'max_age_s','max_camera_skew_s'}:
        raise ValueError('Invalid global conversion policy')
    ages=np.asarray(policy.get('max_age_s'),dtype=float);skew=policy.get('max_camera_skew_s')
    if ages.shape!=(5,) or not np.isfinite(ages).all() or (ages<=0).any() or (skew is not None and (not np.isfinite(skew) or skew<=0)):
        raise ValueError('Invalid global freshness/skew configuration')
    if expected_version not in (None,3,4) or expected_architecture not in (None,'executor_v1','decoupled_teleoperation_v1') or (expected_version==3 and expected_architecture is not None):
        raise ValueError('Invalid global dataset contract')
    episodes = discover(directory)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result=dict(schema_version=1,total_recordings=len(episodes),recordings=[],manifest=None)
    schema=('3.0.0' if expected_version==3 else '4.1.0' if expected_architecture=='decoupled_teleoperation_v1'
            else '4.0.0' if expected_version==4 and expected_architecture=='executor_v1' else None)
    seen = set()
    try:
        existing=_manifest_content(destination,schema)
        if existing['episodes'] or (destination/'manifest.json').exists():schema=existing['schema_version']
        existing_ids={zarr.open_group(str(destination/e['path']),mode='r').attrs['source_episode_id'] for e in existing['episodes']}
        for index, episode in enumerate(episodes, 1):
            report(index,len(episodes),episode.name,'converting')
            entry=dict(path=str(episode),episode_id=None,status=None,total_actions=None,excluded_actions=None,
                       exclusions=None,valid_transitions=None,segments=None,failure=None)
            result['recordings'].append(entry)
            try:
                # Input metadata errors are local; permission and destination I/O errors are not.
                try:
                    metadata=json.loads((episode/'episode.json').read_text())
                    identity=metadata['bfm']['episode']['episode_id']
                    entry['episode_id']=identity if isinstance(identity,str) else None
                    metadata,commit,name,key=source_identity(episode,policy)
                except (KeyError,TypeError,yaml.YAMLError) as exc:raise ValueError(f'Invalid recording metadata: {exc}') from exc
                if expected_version is not None and metadata['bfm']['contract_version']!=expected_version:
                    raise ValueError('Requested dataset contract differs from recorded source')
                if expected_architecture is not None and metadata['bfm'].get('control_architecture','executor_v1')!=expected_architecture:
                    raise ValueError('Requested control architecture differs from recorded source')
                identity=metadata['bfm']['episode']['episode_id'];entry['episode_id']=identity
                if identity in seen:raise BatchCompatibilityError('Duplicate raw episode identity: '+identity)
                seen.add(identity)
                if metadata['bfm']['contract_version'] not in (3,4) or metadata['bfm'].get('control_architecture','executor_v1') not in ('executor_v1','decoupled_teleoperation_v1'):
                    raise ValueError('Unsupported recording contract')
                source_schema=('3.0.0' if metadata['bfm']['contract_version']==3 else
                               '4.1.0' if metadata['bfm'].get('control_architecture')=='decoupled_teleoperation_v1' else '4.0.0')
                if schema is not None and source_schema!=schema:raise BatchCompatibilityError('Mixed destination dataset contracts')
                output=destination/name
                if output.exists():
                    root=zarr.open_group(str(output),mode='r');stats=validate(root)
                    if root.attrs.get('conversion_key')!=key or root.attrs['source_metadata']!=metadata:
                        raise BatchCompatibilityError(f'Existing output differs from raw source/policy/converter; move it aside explicitly: {output}')
                    state='skipped';status='REUSED'
                else:
                    with tempfile.TemporaryDirectory(prefix='.conversion-',dir=destination) as scratch:
                        staged=Path(scratch)/name
                        with tempfile.TemporaryFile(mode='w+') as diagnostics, redirect_stdout(diagnostics):
                            convert(episode,staged,collection_commit=commit,**policy)
                        root=zarr.open_group(str(staged),mode='a');root.attrs['conversion_key']=key;stats=validate(root)
                        if output.exists():raise FileExistsError(output)
                        os.rename(staged,output)
                    root=zarr.open_group(str(output),mode='r')
                    state='converted_with_skips' if root.attrs.get('rejected_ticks',0) else 'validated'
                    status='CONVERTED_WITH_SKIPS' if state=='converted_with_skips' else 'CONVERTED'
                schema=source_schema
                entry.update(status=status,total_actions=root.attrs.get('total_action_ticks'),excluded_actions=root.attrs.get('rejected_ticks'),
                             exclusions=root.attrs.get('rejected_decisions'),valid_transitions=stats['transitions'],segments=stats['segments'])
            except (ValueError,McapError,FileNotFoundError) as exc:
                if isinstance(exc,BatchCompatibilityError):raise
                if isinstance(exc,FileNotFoundError) and (not exc.filename or Path(exc.filename).resolve().is_relative_to(destination) or not Path(exc.filename).resolve().is_relative_to(episode)):
                    raise
                if entry['episode_id'] in existing_ids:
                    raise BatchCompatibilityError(f'{episode}: failed source already represented in destination; existing manifest preserved: {exc}') from exc
                entry.update(status='FAILED',failure=dict(exception_type=type(exc).__name__,message=str(exc)))
                entry.update(getattr(exc,'conversion_details',{}))
                if entry['total_actions'] is None:entry['total_actions']=getattr(exc,'total_action_ticks',None)
                state='failed'
            if report is progress:report(index,len(episodes),episode.name,state,transitions=entry['valid_transitions'] if state!='failed' else None,error=entry['failure']['message'] if entry['failure'] else None)
            else:report(index,len(episodes),episode.name,state)
        if not list(destination.glob('*.zarr')):
            result['completed']=True
            raise NoSuccessfulRecordings('No successfully converted recordings; see conversion_report.json')
        manifest=manifest_for(destination)
        result.update(manifest=str(manifest),completed=True)
        return manifest
    except BaseException as exc:
        if not isinstance(exc,NoSuccessfulRecordings):
            result['aborted']=dict(exception_type=type(exc).__name__,message=str(exc))
            if result['recordings'] and result['recordings'][-1]['status'] is None:
                result['recordings'][-1].update(status='FAILED',failure=result['aborted'])
        raise
    finally:_write_report(destination,result,report)


def _manifest_content(destination, schema=None):
    destination = Path(destination)
    manifest = destination/'manifest.json'
    existing = json.loads(manifest.read_text()) if manifest.exists() else {}
    if manifest.exists() and (not isinstance(existing,dict) or existing.get('schema_version') not in ('3.0.0','4.0.0','4.1.0')
            or not existing.get('dataset_id') or not isinstance(existing.get('episodes'),list)):
        raise ValueError('Invalid existing dataset manifest')
    assignments = {}
    for entry in existing.get('episodes', []):
        path = (destination/entry['path']).resolve()
        if path in assignments or entry['split'] not in ('train', 'validation', 'test') or not path.exists():
            raise ValueError('Invalid or missing existing split assignment')
        assignments[path] = entry['split']
    entries = [];version=existing.get("schema_version",schema);identities=set()
    if schema is not None and version!=schema:raise BatchCompatibilityError('Destination manifest contract mismatch')
    for path in sorted(destination.glob('*.zarr')):
        root=zarr.open_group(str(path), mode='r')
        validate(root)
        identity=root.attrs['source_episode_id']
        if identity in identities:raise ValueError('Duplicate converted episode identity')
        identities.add(identity)
        if version is None:version=root.attrs['schema_version']
        if version!=root.attrs['schema_version']:raise ValueError('Mixed dataset schemas')
        entries.append({'path':path.name, 'split':assignments.get(path.resolve(), 'train')})
    if set(assignments) - {p.resolve() for p in destination.glob('*.zarr')}:
        raise ValueError('Existing manifest references stores outside canonical data/zarr')
    content = {'schema_version':version or '4.0.0', 'dataset_id':existing.get('dataset_id', 'recordings-sharpa-decoupled-v4' if version=='4.1.0' else 'recordings-sharpa-v4' if version=='4.0.0' else 'recordings-sharpa-v3'), 'episodes':entries}
    return content


def manifest_for(destination):
    destination=Path(destination);manifest=destination/'manifest.json'
    content=_manifest_content(destination)
    temporary = destination/f'.manifest-{uuid.uuid4().hex}.json'
    try:
        temporary.write_text(json.dumps(content, indent=2)+'\n')
        SharpaDataset(temporary, observation_profile='vision_proprio_v1')
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest
