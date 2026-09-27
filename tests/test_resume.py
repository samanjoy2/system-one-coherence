import json
import pytest
from coherence import run as runner
from coherence import backends
from coherence.core import save_json


def test_resume_makes_no_duplicate_model_calls(tmp_path,monkeypatch):
    tax = {'d':{'parents':{'P':'Parent'},'children':{'a':'A','b':'B'},'child_parent':{'a':'P','b':'P'}}}
    examples = [{'id':'d:1','dataset':'d','text':'hello','gold_parent':'P','gold_child':'a'}]
    save_json(tmp_path/'data/examples.json',examples)
    save_json(tmp_path/'data/taxonomies.json',tax)
    save_json(tmp_path/'data/manifest.json',{})
    calls = []
    class Fake:
        meta = {'model':'fake','precision':'test'}
        def predict(self,task):
            calls.append(task['name'])
            return {key:1/len(task['criteria']) for key in task['criteria']},{}
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    runner.run('laya',instance=Fake())
    runner.run('laya',instance=Fake())
    assert calls==['coarse','flat','conditional:P']
    assert len(list((tmp_path/'results/laya').glob('*/records/*.json')))==1
    selected = json.loads((tmp_path/'results/primary_runs.json').read_text())
    class Failed:
        meta = {'model':'failed-preflight','precision':'test'}
        def predict(self,task):
            raise ValueError('preflight rejected')
    with pytest.raises(ValueError,match='preflight rejected'):
        runner.run('laya',instance=Failed())
    assert json.loads((tmp_path/'results/primary_runs.json').read_text()) == selected
