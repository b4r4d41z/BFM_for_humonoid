"""Regressions for simulation policy snapshots and explicit recording lifecycle."""
import copy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import zarr

from test_sharpa import source, make
from bfm.data.mcap import ACTION, HELD, RAW_ARM, EPISODE, RECORDING, CLOCK
from bfm.data.sharpa import write, validate


@pytest.fixture
def simulation_source(source):
    records, metadata = copy.deepcopy(source)
    metadata['bfm'].update(policy_dt=.1, policy_clock={
        'domain': 'isaac_simulation', 'topic': CLOCK, 'period_ns': 100_000_000,
        'quantization': 'first physics clock sample at or after deadline'})
    for i, (_, msg) in enumerate(records[ACTION]):
        msg.header.stamp = NS(sec=0, nanosec=i*100_000_000)
    return records, metadata


def stretch_wall_time(records):
    origin = records[ACTION][0][0]
    wall = [0, 320, 650, 960, 1300, 1800, 2110, 2500, 2890, 3250]
    for topic, samples in records.items():
        records[topic] = [(origin + wall[i]*1_000_000 + (t-origin-i*100_000_000), m)
                          for i, (t, m) in enumerate(samples)]


def event(time, kind, identity='generated', **extra):
    value = {'event_schema_version': 1, 'event_type': kind,
             'episode': {'episode_id': identity}, **extra}
    return time, NS(data=json.dumps(value))


def test_simulation_adjacency_with_slow_variable_wall_time(simulation_source, tmp_path):
    records, meta = simulation_source
    stretch_wall_time(records)
    arrays, attrs = make((records, meta))
    assert len(arrays['action']) == 9 and attrs['rejected_ticks'] == 0
    assert attrs['policy_timing']['period_ns'] == 100_000_000
    assert np.allclose(arrays['next_timestamp']-arrays['timestamp'], .1)
    assert np.all(arrays['next_record_ns']-arrays['record_ns'] > 300_000_000)
    path = tmp_path/'slow.zarr'
    write(path, arrays, attrs)
    root = zarr.open_group(str(path), mode='a')
    assert validate(root)['transitions'] == 9
    root['next_action_header_ns'][0] += 100_000_000
    with pytest.raises(ValueError, match='simulation policy adjacency'):
        validate(root)


@pytest.mark.parametrize('kind', ['duplicate', 'backward', 'missing', 'missing_header', 'negative', 'overflow'])
def test_invalid_simulation_stamps_reject_and_split(simulation_source, kind):
    records, meta = simulation_source
    msg = records[ACTION][4][1]
    if kind == 'duplicate': msg.header.stamp.nanosec = 300_000_000
    if kind == 'backward': msg.header.stamp.nanosec = 200_000_000
    if kind == 'missing': del msg.header.stamp
    if kind == 'missing_header': del msg.header
    if kind == 'negative': msg.header.stamp.sec = -1
    if kind == 'overflow': msg.header.stamp.nanosec = 1_000_000_000
    arrays, attrs = make((records, meta))
    reason = {'duplicate':'duplicate_policy_stamp','backward':'backward_policy_stamp'}.get(kind,'invalid_policy_stamp')
    assert attrs['rejected_decisions'][0]['index'] == 4
    assert reason in attrs['rejected_decisions'][0]['reasons']
    time = records[ACTION][4][0]
    assert time not in arrays['record_ns'] and time not in arrays['next_record_ns']
    assert arrays['truncated'].sum() == 2


def test_skipped_simulation_interval_and_bursts_split(simulation_source):
    records, meta = simulation_source
    for _, msg in records[ACTION][4:]:
        stamp = msg.header.stamp.nanosec + 100_000_000
        msg.header.stamp = NS(sec=stamp//10**9, nanosec=stamp%10**9)
    arrays, attrs = make((records, meta))
    assert attrs['valid_decision_ticks'] == 10 and attrs['rejected_ticks'] == 0
    assert attrs['policy_interval_reasons'] == {'skipped_policy_interval': 1}
    assert len(arrays['action']) == 8 and arrays['truncated'].sum() == 2
    records[ACTION][2][1].header.stamp.nanosec = 150_000_000
    _, attrs = make((records, meta))
    assert attrs['policy_interval_reasons']['policy_burst'] == 1


def test_only_observed_physics_step_quantization_is_allowed(simulation_source):
    records, meta = simulation_source
    records[CLOCK] = [(i, NS(clock=NS(sec=0, nanosec=round(i*1e9/60)))) for i in range(20)]
    records[ACTION][2][1].header.stamp.nanosec += 16_666_667
    arrays, attrs = make((records, meta))
    assert len(arrays['action']) == 9
    assert attrs['policy_timing']['tolerance_ns'] == 16_666_668
    records.pop(CLOCK)
    assert len(make((records, meta))[0]['action']) == 7


def test_explicit_lifecycle_without_phase_and_no_required_result(simulation_source):
    records, meta = simulation_source
    t = records[ACTION][0][0]
    records[EPISODE] = [event(t+90_000_000, 'recording_started'),
                        event(t+450_000_000, 'recording_end_requested'),
                        event(t+690_000_000, 'recording_started')]
    arrays, attrs = make((records, meta))
    assert attrs['valid_decision_ticks'] == 7
    assert [r['index'] for r in attrs['rejected_decisions']] == [0,5,6]
    assert len(arrays['action']) == 5 and arrays['truncated'].sum() == 2
    assert not arrays['terminated'].any()


@pytest.mark.parametrize('boundary', ['explicit', 'bool', 'identity', 'legacy', 'legacy_boundary_id'])
def test_never_stitch_through_boundary_between_active_ticks(simulation_source, boundary):
    records, meta = simulation_source
    t = records[ACTION][0][0]
    records[EPISODE] = [event(t-1, 'recording_started')]
    if boundary == 'explicit':
        records[EPISODE] += [event(t+440_000_000,'recording_end_requested'), event(t+460_000_000,'recording_started')]
    if boundary == 'identity':
        records[EPISODE] += [event(t+440_000_000,'recording_started','other'), event(t+460_000_000,'recording_started')]
    if boundary == 'bool':
        records[RECORDING] = [(t-1,NS(data=True)),(t+440_000_000,NS(data=False)),(t+460_000_000,NS(data=True))]
    if boundary == 'legacy':
        records[EPISODE] = [(time, NS(data=json.dumps({'phase':phase,'episode':{'episode_id':'generated'}})))
                            for time,phase in [(t-1,'playing'),(t+440_000_000,'paused'),(t+460_000_000,'playing')]]
    if boundary == 'legacy_boundary_id':
        records[EPISODE] = [(time, NS(data=json.dumps({'phase':'playing', 'boundary_id':identity,
                            'episode':{'episode_id':'generated'}})))
                            for time,identity in [(t-1,'first'),(t+450_000_000,'second')]]
    arrays, attrs = make((records, meta))
    assert attrs['valid_decision_ticks'] == 10
    assert len(arrays['action']) == 8 and arrays['truncated'].sum() == 2
    assert t+400_000_000 not in arrays['record_ns']


def test_end_is_not_reopened_by_recording_bool_and_final_json_wins(simulation_source):
    records, meta = simulation_source
    t = records[ACTION][0][0]
    meta['success'] = False
    records[EPISODE] = [event(t-1,'recording_started',success=True), event(t+450_000_000,'recording_result',success=True)]
    records[RECORDING] = [(t-1,NS(data=True)),(t+500_000_000,NS(data=True))]
    arrays, attrs = make((records, meta))
    assert attrs['success'] is False and attrs['valid_decision_ticks'] == 5
    assert len(arrays['action']) == 4


def high_rate(records):
    for topic in HELD:
        records[topic] = [(t-5_000, m) for t, m in records[topic]]
        extra = []
        for t, msg in records[topic]:
            for offset in (-40_000_000,-20_000_000):
                other = copy.deepcopy(msg)
                if topic == HELD[0]: other.position[0] += .2
                else: other.left_hand_position[0] += 1
                extra.append((t+offset,other))
        records[topic] += extra


def test_high_rate_commands_validate_latest_target_and_raw_source(simulation_source):
    records, meta = simulation_source
    high_rate(records)
    arrays, attrs = make((records, meta))
    assert len(records[HELD[0]]) == 3*len(records[ACTION])
    assert len(arrays['action']) == 9
    assert attrs['held_command_checks'][HELD[0]] == 10
    assert attrs['raw_command_checks']['count'] == 10


@pytest.mark.parametrize('fault', ['newer_unrelated','raw_header','raw_value'])
def test_unrelated_command_cannot_validate_snapshot(simulation_source, fault):
    records, meta = simulation_source
    high_rate(records)
    time, command = records[HELD[0]][4]
    if fault == 'newer_unrelated':
        other = copy.deepcopy(command); other.position[0] += .2
        records[HELD[0]].append((time+100,other))
        records[ACTION][4] = (time+200, records[ACTION][4][1])
    if fault == 'raw_header': records[RAW_ARM][4][1].header.stamp.nanosec += 100
    if fault == 'raw_value': records[RAW_ARM][4][1].position[0] += .2
    with pytest.raises(ValueError, match='Held command|Raw arm provenance'):
        make((records,meta))


def test_legacy_v3_wall_clock_still_supported(source):
    arrays, attrs = make(source)
    assert len(arrays['action']) == 9
    assert attrs['policy_timing']['domain'] == 'mcap_log_time_ns'


@pytest.mark.parametrize('fault', ['period','domain','lifecycle_version','episode_id','recording_commit'])
def test_malformed_contract_is_not_silently_accepted(simulation_source, fault):
    records, meta = simulation_source
    if fault == 'period': meta['bfm']['policy_clock']['period_ns'] = 200_000_000
    if fault == 'domain': meta['bfm']['policy_clock']['domain'] = 'unknown'
    if fault == 'lifecycle_version':
        records[EPISODE] = [event(records[ACTION][0][0]-1,'recording_started',event_schema_version=2)]
    if fault == 'episode_id':
        records[EPISODE] = [event(records[ACTION][0][0]-1,'recording_started',identity='other')]
    if fault == 'recording_commit': meta['recording_code'] = {'commit_sha':'f'*40}
    with pytest.raises(ValueError):
        make((records,meta))
