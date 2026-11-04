from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from bfm.fb.frequency_analysis import spectrum, analyze_frequency
from bfm.fb.runtime import checkpoint_digest
from bfm.fb.spectral_comparison import compare
from bfm.fb.graph_spectral_analysis import analyze_graph
from bfm.fb.spectral_utils import agreement, assignments


def test_time_spectrum():
    grid = np.linspace(0, 5, 501)
    for n in (200, 400):
        t = np.arange(n)*.1; b = np.sin(2*np.pi*1.2*t)[:,None]*np.array([[1.,2.,3.]])
        p, energy = spectrum(t,b,grid)
        assert grid[p.argmax()] == pytest.approx(1.2,abs=.04)
        np.testing.assert_allclose(p,spectrum(t,b[:,::-1],grid)[0])
        assert energy > 0 and np.isfinite(p).all()
    p, energy = spectrum(t,np.ones_like(b),grid)
    assert energy == 0 and not p.any()
    for bad in (t[::-1], np.zeros(len(t)), np.full(len(t),np.nan)):
        with pytest.raises(ValueError): spectrum(bad,b,grid)
    with pytest.raises(ValueError): spectrum(t,np.full_like(b,np.nan),grid)
    gap = t.copy(); gap[len(gap)//2:] += 5
    with pytest.raises(ValueError, match="gap"): spectrum(gap,b,grid)


def test_alignment_metrics():
    assert assignments([dict(episode_id='b',cluster_id=0),dict(episode_id='a',cluster_id=1)],['a','b']) == [1,0]
    for rows in ([dict(episode_id='a',cluster_id=0)]*2, []):
        with pytest.raises(ValueError): assignments(rows,['a','b'])
    result = agreement([0,0,1,1],[1,1,0,0])
    assert result['ari'] == pytest.approx(1) and result['nmi'] == pytest.approx(1)
    result = agreement([0,0,1,1],[0,1,0,1])
    assert result['ari'] == pytest.approx(-.5) and result['nmi'] == pytest.approx(0)


def test_synthetic_workflow(tmp_path,monkeypatch):
    import bfm.fb.frequency_analysis as module
    t = np.arange(100)*.1; ids = ['a','b','c','d']
    z = torch.tensor([[2.,0,0,0],[1.99,.1,0,0],[-2.,0,0,0],[-1.99,-.1,0,0]])
    z = z/z.norm(dim=1,keepdim=True)*2
    sources = [dict(episode_id=e,path=e,source_fingerprint=e,states=100) for e in ids]
    artifact = tmp_path/'z.pt'
    torch.save(dict(format='bfm.episode_z.v1',checkpoint_sha256='test',latent_dim=4,
                    episodes=sources,episode_z=z,global_z=z[:1]),artifact)
    before = artifact.read_bytes()
    runtime = SimpleNamespace(checkpoint_sha256='test', model=SimpleNamespace(cfg=SimpleNamespace(latent_dim=4)))
    episodes = [dict(episode_id=e,path=e,success=True,embeddings=torch.tensor(np.sin(2*np.pi*f*t)[:,None]*np.ones((1,4))))
                for e,f in zip(ids,[.5,.5,2.,2.])]
    monkeypatch.setattr(module,'episode_embeddings',lambda *args:episodes)
    def dataset(e):
        root={'record_ns':np.arange(99),'next_record_ns':np.arange(1,100),
              'timestamp':t[:-1],'next_timestamp':t[1:],'segment_id':np.zeros(99,dtype=int)}
        return SimpleNamespace(roots=[root],episode_fingerprints=[e])
    monkeypatch.setattr(module,'source_dataset',dataset)
    f = analyze_frequency(runtime,'unused',artifact,tmp_path/'f')
    assert f['selected_k']==2
    g = analyze_graph(artifact,tmp_path/'g',neighbors=[1,2],seeds=[0,1])
    assert len(g['studies'])==2
    c = tmp_path/'c';c.mkdir()
    (c/'cluster_summary.json').write_text(json.dumps(dict(checkpoint_sha256='test',analysis_sha256=checkpoint_digest(artifact))))
    (c/'clusters.csv').write_text('episode_id,cluster_id\na,0\nb,0\nc,1\nd,1\n')
    result = compare(artifact,c,tmp_path/'f',tmp_path/'g',tmp_path/'comparison')
    assert result['agreement']['cosine_frequency']['ari']==pytest.approx(1)
    assert (tmp_path/'comparison/cluster_comparison.svg').exists()
    assert artifact.read_bytes()==before
    (c/'cluster_summary.json').write_text(json.dumps(dict(checkpoint_sha256='wrong',analysis_sha256=checkpoint_digest(artifact))))
    with pytest.raises(ValueError,match='mismatch'): compare(artifact,c,tmp_path/'f',tmp_path/'g',tmp_path/'bad')
