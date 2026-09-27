import numpy as np
import pandas as pd
from .core import ROOT,read_json,save_json
from .scope import CATEGORICAL_MODELS


def audit_report():
    primary = {}
    runs = []
    selection_path = ROOT/'results/primary_runs.json'
    selection = read_json(selection_path) if selection_path.exists() else {}
    for path in (ROOT/"results").glob("*/*/manifest.json"):
        m = read_json(path)
        model = path.parent.parent.name
        if model not in CATEGORICAL_MODELS: continue
        rows = {r['id']:r for f in (path.parent/"records").glob("*.json") for r in [read_json(f)]}
        if not rows: continue
        if m['permutation']==0 and m['repeat']==0 and not (model in ('kev','qwen') and m['backend']['precision']=='fp32'):
            if model in selection and path.parent.name!=selection[model]: continue
            if model in primary: raise ValueError("Ambiguous primary run")
            primary[model] = rows
        else: runs.append((model,m,rows))
    comparisons = []
    for model,m,rows in runs:
        if model not in primary: continue
        kind = 'precision' if m['backend']['precision']=='fp32' and model in ('kev','qwen') else ('repeat' if m['repeat'] else 'order')
        for id,row in rows.items():
            if id not in primary[model]: continue
            original = primary[model][id]
            for structure in ['coarse','flat','marginal']:
                p = np.array(original['metrics'][structure]); q = np.array(row['metrics'][structure])
                comparisons.append({'model':model,'dataset':row['dataset'],'id':id,'audit':kind,
                                    'permutation':m['permutation'],'repeat':m['repeat'],'structure':structure,
                                    'tv':np.abs(p-q).sum()/2,'mean_absolute_error':np.abs(p-q).mean(),
                                    'maximum_error':np.abs(p-q).max(),'prediction_flip':int(p.argmax()!=q.argmax()),
                                    'exact_match':bool(np.array_equal(p,q))})
    output = ROOT/'results/report'
    output.mkdir(parents=True,exist_ok=True)
    if comparisons:
        df = pd.DataFrame(comparisons)
        df.to_csv(output/'audit_per_example.csv',index=False)
        df.groupby(['model','dataset','audit','structure']).agg(n=('id','size'),tv=('tv','mean'),
            mean_absolute_error=('mean_absolute_error','mean'),maximum_error=('maximum_error','max'),
            prediction_flip=('prediction_flip','mean'),exact_match=('exact_match','mean')).to_csv(output/'audit_summary.csv')
    save_json(output/'audit_coverage.json',{'comparisons':len(comparisons),
              'runs':[{'model':name,'permutation':m['permutation'],'repeat':m['repeat'],
                       'precision':m['backend']['precision'],'examples':len(rows)} for name,m,rows in runs]})


if __name__=='__main__': audit_report()
