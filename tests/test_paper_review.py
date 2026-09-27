import numpy as np
import pytest
from coherence.paper_review import tie_summary
from coherence.core import derive
from coherence.analysis import ece


def test_tie_bounds_include_deterministic_accuracy():
    r = tie_summary([[.5, .5], [.2, .8], [.5, .5]], [0, 1, 1])
    assert r['ties'] == 2
    assert r['tie_accuracy_low'] == pytest.approx(1/3)
    assert r['accuracy'] == pytest.approx(2/3)
    assert r['tie_accuracy_high'] == 1


def test_parent_discrepancy_contracts_fine_discrepancy():
    tax = {'parents': {'A':'A','B':'B'}, 'children': {'a':'a','b':'b','c':'c'},
           'child_parent': {'a':'A','b':'A','c':'B'}}
    example = {'gold_child':'a','gold_parent':'A'}
    r = derive(example, tax, {'coarse': {'A':.3,'B':.7},
        'flat': {'a':.7,'b':.1,'c':.2},
        'conditional:A': {'a':.8,'b':.2},'conditional:B': {'c':1.}})
    assert r['coarse_tv'] <= np.abs(np.array(r['flat'])-r['marginal']).sum()/2 + 1e-12


def test_ece_boundary_and_unit_confidence():
    p = np.array([[.5,.5],[0.,1.]])
    for bins in (10,15,20):
        assert ece(np.array([0,1]),p,bins) == pytest.approx(.25)
