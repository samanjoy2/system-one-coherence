import numpy as np
from .core import ROOT, SEED, save_json


def audit_examples(rows,taxonomies):
    rng = np.random.default_rng(SEED+1)
    result = []
    for dataset,tax in sorted(taxonomies.items()):
        parents = sorted(tax["parents"])
        quotas = {p:30//len(parents) for p in parents}
        for p in rng.permutation(parents)[:30%len(parents)]: quotas[p] += 1
        for p in parents:
            options = sorted([r for r in rows if r["dataset"]==dataset and r["gold_parent"]==p],key=lambda r:r["id"])
            result.extend(options[i] for i in rng.choice(len(options),quotas[p],replace=False))
    save_json(ROOT/"data/audit_ids.json",[e["id"] for e in result])
    groups = [[e for e in result if e["dataset"]==d] for d in sorted(taxonomies)]
    for group in groups: rng.shuffle(group)
    return [g[i] for i in range(30) for g in groups]


def run_audit(model_name,precision=False):
    from .backends import backend
    from .run import run
    model = backend(model_name,"fp32" if precision else "int8")
    if precision:
        run(model_name,limit=30,audit=True,precision="fp32",instance=model)
    else:
        for permutation in range(1,5):
            run(model_name,audit=True,permutation=permutation,instance=model)
        if model_name=="jev":
            for repeat in [1,2]: run(model_name,audit=True,repeat=repeat,instance=model)
