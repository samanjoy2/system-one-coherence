"""Audit dataset integrity and completed-run coverage without making model calls."""
from collections import Counter
import numpy as np
from .core import ROOT,read_json,save_json,digest,distribution,derive


def validate():
    examples = read_json(ROOT/'data/examples.json')
    taxonomies = read_json(ROOT/'data/taxonomies.json')
    manifest = read_json(ROOT/'data/manifest.json')
    assert len(examples)==2500 and len({e['id'] for e in examples})==2500
    assert digest(examples)==manifest['examples_hash'] and digest(taxonomies)==manifest['taxonomy_hash']
    datasets = {}
    for d,tax in taxonomies.items():
        rows = [e for e in examples if e['dataset']==d]
        assert set(tax['children'])==set(tax['child_parent'])
        assert set(tax['child_parent'].values())==set(tax['parents'])
        for e in rows: assert tax['child_parent'][e['gold_child']]==e['gold_parent']
        counts = Counter(e['gold_child'] for e in rows)
        datasets[d] = {'n':len(rows),'represented_children':len(counts),'declared_children':len(tax['children']),
                       'minimum_stratum':min(counts.values()),'maximum_stratum':max(counts.values()),
                       'duplicate_text_count':len(rows)-len({e['text'] for e in rows}),
                       'absent_test_labels':sorted(set(tax['children'])-set(counts))}
    runs = []
    originals = {e['id']:e for e in examples}
    for path in (ROOT/'results').glob('*/*/manifest.json'):
        meta = read_json(path)
        counts = Counter()
        for file in (path.parent/'records').glob('*.json'):
            row = read_json(file)
            assert all(row[k]==v for k,v in originals[row['id']].items()), 'Result example differs from frozen dataset'
            t = taxonomies[row['dataset']]
            recomputed = derive(row,t,row['probabilities'])
            for key,value in recomputed.items():
                assert np.allclose(row['metrics'][key],value,rtol=1e-10,atol=1e-12), f'Derived metric mismatch: {key}'
            for structure in ['coarse','flat','marginal']:
                labels = sorted(t['parents'] if structure=='coarse' else t['children'])
                distribution(dict(zip(labels,row['metrics'][structure])),labels)
            counts[row['dataset']] += 1
        runs.append({'model':path.parent.parent.name,'run':path.parent.name,'permutation':meta['permutation'],
                     'repeat':meta['repeat'],'precision':meta['backend']['precision'],'counts':dict(counts)})
    report = {'datasets':datasets,'runs':runs}
    save_json(ROOT/'results/validation.json',report)
    print(report,flush=True)


if __name__=='__main__': validate()
