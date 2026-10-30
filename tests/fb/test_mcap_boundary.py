"""Recording joins volatile streams; missing prefixes are not valid actions."""
import copy

import pytest

from test_sharpa import source, make
from bfm.data.mcap import ACTION, HELD, RAW_ARM, TOPICS


@pytest.fixture
def callback_source(source):
    records, metadata = copy.deepcopy(source)
    metadata['bfm']['arm_target_processing'].update(
        controller_cadence='validated source callback rate',
        canonical_action_cadence='0.1 simulation seconds; latest bounded target snapshot')
    for topic in HELD:
        records[topic] = [(t-2_000_000, m) for t, m in records[topic]]
    records[RAW_ARM] = [(t+91_850_000, m) for t, m in records[RAW_ARM]]
    return records, metadata


@pytest.mark.parametrize('prefix', [1, 3])
@pytest.mark.parametrize('missing_topic', [HELD[0], HELD[1], RAW_ARM])
def test_incomplete_leading_chain_is_reported_and_never_bridged(callback_source, prefix, missing_topic):
    records, metadata = callback_source
    records[missing_topic] = records[missing_topic][prefix:]
    arrays, attrs = make((records, metadata))
    rejects = attrs['rejected_decisions']
    assert [r['index'] for r in rejects] == list(range(prefix))
    assert all(r['boundary'] == 'startup' for r in rejects)
    assert all(f'missing_startup_command_provenance:{missing_topic}' in r['reasons'] for r in rejects)
    assert attrs['startup_provenance_rejected_ticks'] == prefix
    assert attrs['valid_decision_ticks'] == 10-prefix
    assert len(arrays['action']) == 9-prefix
    assert arrays['record_ns'][0] == records[ACTION][prefix][0]
    assert attrs['held_command_checks'][HELD[0]] == 10-prefix
    assert attrs['rejected_decisions'] == make((records, metadata))[1]['rejected_decisions']


def test_first_action_has_no_matching_recorded_target(callback_source):
    records, metadata = callback_source
    t = records[ACTION][0][0]
    # Next request is genuinely different, not a delayed copy of action zero.
    # This mirrors the recording-boundary mechanism, not its measured times.
    records[ACTION][0][1].left_arm_target[0] += .2
    for topic in (RAW_ARM, HELD[0]):
        records[topic][0] = (t+3_000_000+(topic == HELD[0]), records[topic][0][1])
    arrays, attrs = make((records, metadata))
    reject = attrs['rejected_decisions'][0]
    assert reject['index'] == 0 and reject['record_ns'] == t
    assert reject['action_header_ns'] >= 0 and reject['boundary'] == 'startup'
    assert len(reject['reasons']) == 2 and len(arrays['action']) == 8
    assert t not in arrays['record_ns'] and t not in arrays['next_record_ns']


@pytest.mark.parametrize('offset_ns', [1_000, 7_000_000, 49_000_000])
def test_action_before_first_raw_and_held_requires_complete_chain(callback_source, offset_ns):
    records, metadata = callback_source
    t = records[ACTION][0][0]
    for topic in (RAW_ARM, HELD[0]):
        records[topic][0] = (t+offset_ns+(topic == HELD[0]), records[topic][0][1])
    arrays, attrs = make((records, metadata))
    assert len(arrays['action']) == 9 and attrs['rejected_ticks'] == 0
    assert attrs['command_provenance'][0]['raw_record_ns'] > t
    assert attrs['command_provenance'][0]['controller_record_ns'] > t
    assert (arrays['obs/source_record_ns'] <= arrays['record_ns'][:, None]).all()
    assert attrs['command_reorder_window_ns'] == 50_000_000


@pytest.mark.parametrize('fault', ['held', 'raw', 'header', 'missing_held', 'missing_raw', 'future_raw', 'delayed', 'next_action', 'conflicting_successor'])
def test_interior_provenance_mismatch_remains_fatal(callback_source, fault):
    records, metadata = callback_source
    t = records[ACTION][4][0]
    # Give this policy snapshot a distinct request, so an older target cannot
    # accidentally satisfy the fixture's otherwise constant canonical action.
    for topic in (RAW_ARM, HELD[0]): records[topic][4][1].position[0] += .1
    records[ACTION][4][1].left_arm_target[0] += .1
    if fault == 'held': records[HELD[0]][4][1].position[0] += .2
    if fault == 'raw': records[RAW_ARM][4][1].position[0] += .2
    if fault == 'header': records[RAW_ARM][4][1].header.stamp.nanosec += 100
    if fault == 'missing_held': del records[HELD[0]][4]
    if fault == 'missing_raw': del records[RAW_ARM][4]
    if fault == 'future_raw': records[RAW_ARM][4] = (t+60_000_000, records[RAW_ARM][4][1])
    if fault == 'next_action':
        records[ACTION][5] = (t+5_000_000, records[ACTION][5][1])
        records[HELD[0]][4] = (t+10_000_000, records[HELD[0]][4][1])
    if fault in ('delayed', 'conflicting_successor'):
        records[HELD[0]][4] = (t+60_000_000, records[HELD[0]][4][1])
        if fault == 'conflicting_successor':
            other = copy.deepcopy(records[HELD[0]][4][1]); other.position[0] += .2
            records[HELD[0]].append((t+1_000_000, other))
            records[HELD[0]][4] = (t+10_000_000, records[HELD[0]][4][1])
    with pytest.raises(ValueError, match='Held command|Raw arm provenance'):
        make((records, metadata))


def test_existing_conflicting_chain_at_start_is_not_missing_provenance(callback_source):
    records, metadata = callback_source
    records[RAW_ARM][0][1].position[0] += .2
    with pytest.raises(ValueError, match='Raw arm provenance'):
        make((records, metadata))


def test_observation_rejection_does_not_restart_provenance_prefix(callback_source):
    records, metadata = callback_source
    for topic in TOPICS: records[topic] = records[topic][2:]
    records[RAW_ARM][1][1].position[0] += .2
    with pytest.raises(ValueError, match='Raw arm provenance.*index 1'):
        make((records, metadata))
