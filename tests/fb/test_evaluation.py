"""Evaluation control decisions and logs without ROS or a simulator."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from bfm.contracts.sharpa import ActionNormalizer
from bfm.contracts.vision import CAMERAS, PROPRIO_JOINTS
from bfm.fb.evaluation import Evaluation, RolloutLogger, RuntimeSafetyFilter, ARMS, validate_evaluation
from bfm.sharpa_runtime import VisualObservationBuffer


def physical():
    return ActionNormalizer().physical(torch.zeros(26)).numpy()


def proprio():
    q = np.zeros(116)
    q[:14] = physical()[ARMS]
    return q


def buffer(now=1.):
    result = VisualObservationBuffer(.15, CAMERAS, .03)
    result.receive('joints', NS(name=list(PROPRIO_JOINTS), position=proprio()[:58], velocity=proprio()[58:]), now)
    for c in CAMERAS:
        result.receive(c, np.zeros((2, 3, 3), dtype=np.uint8), now)
    return result


def evaluator(tmp_path, mode='closed-loop', warmup=2, maximum=4, obs=None, runtime=None):
    log = RolloutLogger(tmp_path/'run', mode=mode, checkpoint='model.pt', task_z='task.pt')
    calls = []
    def act(value):
        calls.append(value)
        return torch.ones(26)
    evaluation = Evaluation(runtime or NS(act=act), obs or buffer(), log, mode=mode,
                            warmup_ticks=warmup, max_policy_steps=maximum, clock=lambda:1.)
    return evaluation, calls


def test_warmup_limit_logging_and_previous_applied(tmp_path):
    evaluation, calls = evaluator(tmp_path)
    published = []
    for _ in range(6): evaluation.tick(lambda a: published.append(a.copy()))
    assert len(calls) == 4 and len(published) == 2
    np.testing.assert_allclose(published[0][ARMS], physical()[ARMS]+.1)
    np.testing.assert_allclose(published[1][ARMS], published[0][ARMS]+.1)
    assert np.all(published[0][7:13] == 100)  # Hands are not radians.
    rows = [json.loads(s) for s in (tmp_path/'run/ticks.jsonl').read_text().splitlines()]
    assert [r['published'] for r in rows] == [False, False, True, True]
    assert rows[2]['raw_physical_action'] != rows[2]['applied_physical_action']
    assert rows[2]['safety_intervention'] and rows[2]['limited_dimensions'] == ARMS.tolist()
    summary = json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['termination_reason'] == 'max_policy_steps'
    assert summary['total_policy_steps'] == 4 and summary['warmup_steps'] == 2
    assert summary['published_policy_steps'] == 2
    assert rows[2]['source_ages_s']['head'] == 0


def test_shadow_never_calls_publisher(tmp_path):
    evaluation, calls = evaluator(tmp_path, mode='shadow', warmup=0, maximum=2)
    def forbidden(*args): raise AssertionError('shadow attempted publication or authority check')
    evaluation.tick(forbidden, forbidden); evaluation.tick(forbidden, forbidden)
    assert len(calls) == 2 and evaluation.stopped
    assert evaluation.logger.summary['published_policy_steps'] == 0


def test_safety_small_change_and_bad_actions():
    safety = RuntimeSafetyFilter(.1)
    action = physical(); action[ARMS] += .02
    applied, limited = safety.apply(action, proprio())
    assert not limited
    np.testing.assert_array_equal(action, applied)
    for bad in (np.zeros(25), np.zeros((1,26)), np.full(26, np.nan), np.full(26, np.inf), np.full(26, -np.inf), np.full(26, 999)):
        with pytest.raises(ValueError): safety.apply(bad, proprio())


@pytest.mark.parametrize('kind', ['missing', 'stale', 'latency', 'action', 'inference', 'publisher'])
def test_failures_cannot_publish(tmp_path, kind):
    obs = buffer()
    if kind == 'missing': obs.samples.clear()
    if kind == 'stale': obs.samples['head'] = (.1, obs.samples['head'][1])
    def act(value):
        if kind == 'inference': raise RuntimeError('inference failed')
        if kind == 'latency': obs.samples['head'] = (.1, obs.samples['head'][1])
        return torch.full((26,), float('nan')) if kind == 'action' else torch.zeros(26)
    evaluation, _ = evaluator(tmp_path, warmup=0, maximum=1, obs=obs, runtime=NS(act=act))
    sent = []
    def authority():
        if kind == 'publisher': raise RuntimeError('competing publisher')
    if kind == 'missing': evaluation.tick(sent.append, authority)
    else:
        with pytest.raises((ValueError, RuntimeError)): evaluation.tick(sent.append, authority)
    assert not sent and evaluation.stopped
    summary = json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['published_policy_steps'] == 0
    if kind != 'missing': assert summary['termination_reason'] != 'max_policy_steps'


def test_missing_resets_consecutive_warmup(tmp_path):
    evaluation, _ = evaluator(tmp_path, maximum=5)
    sent = []
    evaluation.tick(sent.append)
    saved = evaluation.buffer.samples.pop('head')
    evaluation.tick(sent.append)
    evaluation.buffer.samples['head'] = saved
    evaluation.tick(sent.append); evaluation.tick(sent.append)
    assert not sent
    evaluation.tick(sent.append)
    assert len(sent) == 1


@pytest.mark.parametrize('options', [('bad',0,1,.1), ('shadow',-1,1,.1), ('shadow',0,0,.1),
                                     ('shadow',0,1,0), ('shadow',0,1,float('nan')), ('shadow',True,1,.1)])
def test_invalid_configuration(options):
    with pytest.raises(ValueError): validate_evaluation(*options)


from test_runtime_tasks import trained, visual_manifest, source


def test_loaded_visual_runtime_offline_rollout(trained, tmp_path):
    runtime, _, _ = trained
    task = tmp_path/'task.pt'
    torch.save({'z': runtime.model.sample_z(1), 'checkpoint_sha256': runtime.checkpoint_sha256}, task)
    runtime.load_task(task, require_bound=True)
    bare = tmp_path/'bare.pt'
    torch.save(runtime.z, bare)
    with pytest.raises(ValueError, match='checkpoint-bound'):
        runtime.load_task(bare, require_bound=True)
    evaluation, _ = evaluator(tmp_path/'evaluation', warmup=0, maximum=1, runtime=runtime)
    sent = []
    evaluation.tick(sent.append)
    assert len(sent) == 1 and sent[0].shape == (26,)
    ActionNormalizer().normalize(sent[0])
    assert evaluation.logger.summary['published_policy_steps'] == 1


@pytest.mark.parametrize('missing', [False, True])
def test_observation_loss_after_publication_stops(tmp_path, missing):
    evaluation, _ = evaluator(tmp_path, warmup=0)
    sent = []
    evaluation.tick(sent.append)
    if missing: evaluation.buffer.samples.clear()
    else: evaluation.buffer.samples['head'] = (.1, evaluation.buffer.samples['head'][1])
    with pytest.raises(RuntimeError): evaluation.tick(sent.append)
    evaluation.tick(sent.append)
    assert len(sent) == 1 and evaluation.stopped
    assert evaluation.logger.summary['invalid_stale_observations'] == 1


def test_callback_failure_summary(tmp_path):
    evaluation, _ = evaluator(tmp_path)
    evaluation.stop('observation callback failed', invalid_observation=True)
    evaluation.tick(lambda a: pytest.fail('Stopped runtime published'))
    assert evaluation.logger.summary['invalid_stale_observations'] == 1
    assert evaluation.logger.summary['termination_reason'] == 'observation callback failed'
