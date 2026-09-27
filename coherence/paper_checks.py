"""Read-only checks of native responses and canonical caches; writes a QA summary."""
from collections import Counter
import numpy as np
from .core import ROOT, read_json, save_json, distribution, digest, questions


def main():
    runs = {'jev':'ca3fd3d5f7377071', 'laya':'a6f8115b3cb88dc5'}
    tax = read_json(ROOT/'data/taxonomies.json')
    examples = {e['id']:e for e in read_json(ROOT/'data/examples.json')}
    summary = {}
    for model, tag in runs.items():
        root = ROOT/'results'/model/tag
        counts = Counter()
        errors = []
        zeros = []
        for example in examples.values():
            row = read_json(root/'records'/(digest(example['id'])+'.json'))
            for task in questions(example,tax[example['dataset']]):
                key = digest({'manifest':tag,'example':example,'task':task})
                cached = read_json(root/'raw'/(key+'.json'))
                assert cached['task'] == task
                native = cached['raw']['response']['answers']['classification']['probabilities']
                labels = list(task['criteria'])
                p,total = distribution(native, labels, decimals=2 if model=='jev' else 4)
                assert np.isclose(total,cached['original_probability_sum'],rtol=0,atol=1e-12)
                assert np.allclose(p,[cached['probabilities'][k] for k in labels],rtol=0,atol=1e-12)
                assert np.allclose(p,[row['probabilities'][task['name']][task['id_label'][k]] for k in labels],rtol=0,atol=1e-12)
                errors.append(abs(total-1))
                if task['name']=='flat': zeros.append(sum(v==0 for v in native.values()))
                counts[example['dataset']]+=1
        summary[model]={'questions':sum(counts.values()),'by_dataset':dict(counts),
                        'nonunit_sums':sum(v>1e-8 for v in errors),'max_mass_error':max(errors),
                        'mean_mass_error':float(np.mean(errors)),
                        'mean_zero_entries_flat':float(np.mean(zeros))}
        assert sum(counts.values())==36000
        print(model,summary[model],flush=True)
    save_json(ROOT/'results/figures/cache_validation.json',summary)


if __name__=='__main__': main()
