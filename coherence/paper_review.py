"""Offline, post-hoc manuscript checks; never queries a model or alters raw records."""
import numpy as np
from .core import ROOT, read_json, save_json
from .analysis import ece
from .paper_assets import records, table, MODELS, DATASETS, NAMES


def tie_summary(probabilities, truth):
    p = np.asarray(probabilities)
    y = np.asarray(truth)
    maxima = p == p.max(axis=1, keepdims=True)
    ties = maxima.sum(axis=1) > 1
    gold_maximal = maxima[np.arange(len(y)), y]
    return {"ties": int(ties.sum()),
            "accuracy": float((p.argmax(axis=1) == y).mean()),
            "tie_accuracy_low": float((gold_maximal & ~ties).mean()),
            "tie_accuracy_high": float(gold_maximal.mean())}


def main():
    groups, tie_rows, ece_rows = [], [], []
    for model in MODELS:
        all_rows = records(model)
        for dataset in DATASETS:
            rows = [r for r in all_rows if r['dataset'] == dataset]
            y = np.array([r['metrics']['gold'] for r in rows])
            entry = {'model': model, 'dataset': dataset, 'n': len(rows)}
            tie_row = [NAMES[dataset], model.title()]
            ece_row = [NAMES[dataset], model.title()]
            for structure in ('flat', 'marginal'):
                p = np.array([r['metrics'][structure] for r in rows])
                summary = tie_summary(p, y)
                summary['ece'] = {str(b): ece(y, p, bins=b) for b in (10, 15, 20)}
                entry[structure] = summary
                tie_row += [summary['ties'],
                            f"{100*summary['tie_accuracy_low']:.1f}--{100*summary['tie_accuracy_high']:.1f}"]
            for b in (10, 15, 20):
                ece_row.append(f"{entry['marginal']['ece'][str(b)]-entry['flat']['ece'][str(b)]:+.3f}")
            # One example per identical text, retaining lexicographically first ID.
            seen, keep = set(), []
            for i, r in enumerate(rows):
                if r['text'] not in seen:
                    seen.add(r['text']); keep.append(i)
            entry['unique_text_n'] = len(keep)
            entry['deduplicated_accuracy_change'] = float(np.mean([
                int(rows[i]['metrics']['marginal_pred'] == y[i]) -
                int(rows[i]['metrics']['flat_pred'] == y[i]) for i in keep]))
            groups.append(entry); tie_rows.append(tie_row); ece_rows.append(ece_row)
    table('tie_sensitivity',
          'Post-hoc sensitivity to exact top-probability ties. Counts use recorded normalized vectors. Accuracy ranges (percent) cover arbitrary choices among tied maxima, holding all non-tied predictions fixed; these are deterministic tie-resolution bounds.',
          'tab:ties', 'llrrrr',
          ['Dataset', 'Model', 'Flat ties', 'Flat range', 'Marg. ties', 'Marg. range'], tie_rows)
    table('ece_sensitivity',
          'Post-hoc bin-count sensitivity: marginal-minus-flat top-label ECE with 10, 15, or 20 equal-width bins. Negative values indicate lower empirical ECE. These descriptive point estimates assess bin-count sensitivity.',
          'tab:ecebins', 'llrrr', ['Dataset', 'Model', '10 bins', '15 bins', '20 bins'], ece_rows)
    save_json(ROOT/'results/figures/review_checks.json', {'post_hoc': True, 'groups': groups})
    print(groups, flush=True)


if __name__ == '__main__':
    main()
