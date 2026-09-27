import numpy as np
import pytest
from sklearn.metrics import f1_score
from coherence.analysis import ece,performance,fast_macro_f1,bootstrap_indices


def test_perfect_and_uniform():
    y = np.array([0,1])
    assert performance(y,np.eye(2)) == {"accuracy":1.,"macro_f1":1.,"brier":0.,"ece":0.}
    assert performance(y,np.full((2,2),.5))["brier"] == .5
    assert ece(y,np.full((2,2),.5)) == 0


def test_bootstrap_strata():
    y = np.array([0,0,1,2,2,2])
    ix = bootstrap_indices(y,reps=20)
    for row in ix:
        assert np.bincount(y[row]).tolist() == [2,1,3]
    assert np.array_equal(ix,bootstrap_indices(y,reps=20))


def test_fast_f1():
    y = np.array([0,0,1,1,2]); pred = np.array([0,1,1,2,2])
    assert fast_macro_f1(y,pred,4) == pytest.approx(f1_score(y,pred,labels=range(4),average='macro',zero_division=0))
