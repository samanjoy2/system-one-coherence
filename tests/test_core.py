import numpy as np
import pytest
from coherence.core import derive, distribution, questions
from coherence.prepare import stratified


TAX = {"parents":{"A":"A","B":"B"},"children":{"a":"a","b":"b","c":"c"},
       "child_parent":{"a":"A","b":"A","c":"B"}}
EX = {"id":"one","text":"text","gold_child":"a","gold_parent":"A"}


def test_hierarchy_and_metrics():
    p = {"coarse":{"A":.6,"B":.4},"flat":{"a":.3,"b":.3,"c":.4},
         "conditional:A":{"a":.5,"b":.5},"conditional:B":{"c":1.}}
    r = derive(EX,TAX,p)
    assert r["cfce"] == 0 and r["jsd"] == 0 and r["drift"] == 0
    assert r["hard_pred"] == 0 and r["marginal_pred"] == 2
    assert r["agreement_hard"] == 0 and r["agreement_marginal"] == 1


def test_disjoint_jsd():
    p = {"coarse":{"A":0.,"B":1.},"flat":{"a":1.,"b":0.,"c":0.},
         "conditional:A":{"a":1.,"b":0.},"conditional:B":{"c":1.}}
    r = derive(EX,TAX,p)
    assert r["jsd"] == 1 and r["drift"] == 1 and r["coarse_tv"] == 1


@pytest.mark.parametrize("raw",[{"a":.2,"b":.2},{"a":float('nan'),"b":1},{"a":-1.,"b":2.}])
def test_invalid_distribution(raw):
    with pytest.raises(ValueError): distribution(raw,["a","b"])


def test_rounding_and_order():
    p,total = distribution({"a":.3333,"b":.3333,"c":.3333},["c","a","b"])
    assert np.isclose(p.sum(),1) and total == pytest.approx(.9999)
    a = questions(EX,TAX)
    assert a == questions(EX,TAX)
    assert a[1]["id_label"] == {k:v for q in a[2:] for k,v in q["id_label"].items()}


def test_rare_stratification():
    rows = [{"id":f"{c}{i}","gold_child":c} for c,n in [("a",1),("b",10),("c",10)] for i in range(n)]
    result = stratified(rows,10,["a","b","c"])
    assert len(result)==10 and len({r['id'] for r in result})==10
    assert sum(r["gold_child"]=="a" for r in result)==1
    assert result == stratified(rows,10,["a","b","c"])
