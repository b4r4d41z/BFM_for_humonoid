import numpy as np
import pytest
from bfm.fb.graph_spectral_analysis import graph, laplacian, kmeans


def test_graph_geometry():
    z=np.array([[1.,.01],[1.,-.01],[-1.,.01],[-1.,-.01]])
    a,_=graph(z,1)
    assert (a>=0).all() and np.allclose(a,a.T) and not np.diag(a).any()
    values,vectors,components,isolated=laplacian(a)
    np.testing.assert_allclose(values,[0,0,2,2],atol=1e-12)
    assert len(components)==2 and isolated==[]
    assert np.argmax(np.diff(values))+1==2
    np.testing.assert_allclose(vectors.T@vectors,np.eye(4),atol=1e-12)
    assert len(laplacian(graph(z,2)[0])[2])==1
    _,_,components,isolated=laplacian(np.zeros((3,3)))
    assert len(components)==3 and isolated==[0,1,2]
    for bad in (0,4):
        with pytest.raises(ValueError):graph(z,bad)
    with pytest.raises(ValueError):laplacian(-np.ones((3,3)))
    np.testing.assert_array_equal(kmeans(z,2,0),kmeans(z,2,0))
    assert len(np.unique(kmeans(np.zeros((5,2)),3,0)))==3
