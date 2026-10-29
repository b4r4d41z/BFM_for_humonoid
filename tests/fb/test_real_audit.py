"""Regressions from the real-sample audit; fixtures isolate indexing/role bugs."""
import copy
import json
import numpy as np
import pytest
import torch
import zarr
from test_sharpa import source, make
from bfm.data.mcap import TOPICS
from bfm.data.sharpa import write, SharpaDataset
from bfm.data.visual_replay import VisualReplayBuffer
from bfm.fb.agent import FBAgent, TrainConfig
from bfm.fb.model import ModelConfig
from bfm.fb.checkpoint import save_checkpoint, load_payload
from bfm.contracts.vision import CAMERAS, PROPRIO_JOINTS


def multi_manifest(source, tmp_path):
    entries=[]
    for store_id, (split, outcome) in enumerate([('validation',False),('train',False),('train',True),('train',False)]):
        records, metadata=copy.deepcopy(source)
        metadata['success']=outcome
        metadata['bfm']['episode']['episode_id']=f'store{store_id}'
        # Reject a middle tick, making two segments in every store.
        records[TOPICS[0]].pop(4)
        for source_topic in TOPICS[1:]:
            for tick, (_, msg) in enumerate(records[source_topic]):
                if source_topic == TOPICS[1]:
                    msg.position=[store_id+tick*.01]*len(msg.name)
                    msg.velocity=[-store_id-tick*.01]*len(msg.name)
                else:
                    camera=TOPICS[2:].index(source_topic)
                    value=store_id*50+tick*3+camera
                    msg.data=bytes([value]*16)
        arrays, attrs=make((records,metadata))
        # Reward sidecar marker, kept continuous in both endpoints.
        for prefix, timekey in [('obs','record_ns'),('next_obs','next_record_ns')]:
            ticks=(arrays[timekey]-arrays['record_ns'][0])//100000000
            arrays[f'{prefix}/state'][:,0]=store_id+ticks*.01
        write(tmp_path/f's{store_id}.zarr',arrays,attrs)
        entries.append({'path':f's{store_id}.zarr','split':split})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version':'3.0.0','dataset_id':'index-audit','episodes':entries}))
    return manifest


def test_multistore_multisegment_filtered_replay_and_references(source,tmp_path):
    manifest=multi_manifest(source,tmp_path)
    data=SharpaDataset(manifest,observation_profile='vision_proprio_v1',outcome='failed')
    assert data.ids==['store1','store3'] and len(data.episodes)==4
    replay=VisualReplayBuffer(data)
    indices=np.arange(replay.offsets[-1])[::-1]  # Unsorted batches expose offset assumptions.
    for prefix,key in [('obs','observation'),('next_obs','next_observation')]:
        batch=replay.observations(indices,prefix)
        assert torch.equal(batch['proprio'],replay.data[key][indices.copy()])
        for batch_row, global_row in enumerate(indices):
            root_id,row=data.image_rows[global_row]
            root=data.roots[root_id]
            assert root.attrs['source_episode_id'] in ('store1','store3')
            q=root[f'{prefix}/joint_position'][row]
            names=root.attrs['joint_names']
            assert batch['proprio'][batch_row,0]==pytest.approx(q[names.index(PROPRIO_JOINTS[0])])
            for camera in CAMERAS:
                assert np.array_equal(batch[camera][batch_row].numpy(),root[f'{prefix}/images/{camera}'][row])
            if prefix=='next_obs':
                assert np.array_equal(replay.data['evaluation_state'][global_row].numpy(),root['next_obs/state'][row])
            assert torch.equal(replay.data['action'][global_row],data.episodes[np.searchsorted(replay.offsets[1:],global_row,side='right')]['action'][global_row-replay.offsets[np.searchsorted(replay.offsets[1:],global_row,side='right')]])
    refs, rows=replay.reference_states(8,17)
    assert torch.equal(refs,replay.data['evaluation_state'][rows])
    cfg=ModelConfig(state_dim=16,action_dim=26,latent_dim=4,hidden_dim=16,
        observation_profile='vision_proprio_v1',vision={'backbone':'mock','image_size':14,'embedding_dim':16,'projection_dim':8})
    agent=FBAgent(cfg,TrainConfig())
    path=tmp_path/'reference.pt'
    save_checkpoint(path,agent,replay,data,refs,rows,{})
    payload=load_payload(path)
    with torch.no_grad():
        exact=agent.model.backward_map(replay.observations(rows,'next_obs')).cpu()
    assert torch.equal(exact,payload['reference_backward'])
    assert torch.equal(refs,payload['reference_states']) and rows==payload['reference_indices']


def test_visual_fingerprint_tracks_rgb_mutation_but_task56_does_not(source,tmp_path):
    manifest=multi_manifest(source,tmp_path)
    before=SharpaDataset(manifest,observation_profile='vision_proprio_v1').fingerprint
    state_before=SharpaDataset(manifest).fingerprint
    root=zarr.open_group(str(tmp_path/'s1.zarr'),mode='a')
    root['obs/images/head'][0]=255  # Keep all source metadata/MCAP hashes unchanged.
    after=SharpaDataset(manifest,observation_profile='vision_proprio_v1').fingerprint
    assert after!=before
    assert SharpaDataset(manifest).fingerprint==state_before


def test_actor_uses_post_fb_fusion_without_another_backbone_pass(source,tmp_path,monkeypatch):
    manifest=multi_manifest(source,tmp_path)
    replay=VisualReplayBuffer(SharpaDataset(manifest,observation_profile='vision_proprio_v1'))
    batch=replay.sample_transitions(4)
    agent=FBAgent(ModelConfig(state_dim=16,action_dim=26,latent_dim=4,hidden_dim=16,
        observation_profile='vision_proprio_v1',vision={'backbone':'mock','image_size':14,'embedding_dim':16,'projection_dim':8}),TrainConfig())
    encoder=agent.model.observation_encoder
    original_features=encoder.features
    calls=[]
    def counted(observation):
        calls.append(observation)
        return original_features(observation)
    monkeypatch.setattr(encoder,'features',counted)
    original_actor=agent.update_actor
    def checked_actor(states,z,**kwargs):
        with torch.no_grad():
            expected=encoder.fusion(agent.model.normalizer(batch['observation']['proprio']),original_features(batch['observation']))
        assert torch.equal(states,expected)
        return original_actor(states,z,**kwargs)
    monkeypatch.setattr(agent,'update_actor',checked_actor)
    agent.update(batch)
    assert len(calls)==2
