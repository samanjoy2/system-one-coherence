"""Rebuild tables and figures from immutable per-example results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .core import ROOT, SEED, read_json, save_json
from .scope import CATEGORICAL_MODELS, EXCLUDED_MODELS, PENDING_MODELS, PRECISION_AUDIT_MODELS


def ece(y,p,bins=15):
    confidence = p.max(axis=1)
    correct = p.argmax(axis=1)==y
    bucket = np.minimum((confidence*bins).astype(int),bins-1)
    # Equivalent to weighted per-bin gaps, without empty-bin division.
    return float(np.abs(np.bincount(bucket,weights=correct.astype(float)-confidence,minlength=bins)).sum()/len(y))


def performance(y,p,pred=None):
    pred = p.argmax(axis=1) if pred is None else pred
    return {"accuracy":float(np.mean(pred==y)),"macro_f1":float(f1_score(y,pred,labels=np.arange(p.shape[1]),zero_division=0,average="macro")),
            "brier":float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1))),"ece":float(ece(y,p))}


def bootstrap_indices(y,reps=10000):
    rng = np.random.default_rng(SEED+2)
    strata = [np.flatnonzero(y==k) for k in np.unique(y)]
    return np.concatenate([rng.choice(idx,size=(reps,len(idx)),replace=True) for idx in strata],axis=1)


def analyze(preview=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifests = []
    records = []
    selection_path = ROOT/'results/primary_runs.json'
    selection = read_json(selection_path) if selection_path.exists() else {}
    for path in sorted((ROOT/"results").glob("*/*/manifest.json")):
        m = read_json(path)
        name = path.parent.parent.name
        if name not in CATEGORICAL_MODELS: continue
        if name in selection and path.parent.name!=selection[name]: continue
        if m["permutation"] or m["repeat"] or m["backend"]["precision"]=="fp32" and path.parent.parent.name in ("kev","qwen"):
            continue
        rows = [read_json(p) for p in sorted((path.parent/"records").glob("*.json"))]
        if rows: manifests.append((path,m,rows))
    # Never silently combine multiple model configurations.
    seen = set()
    for path,m,rows in manifests:
        name = path.parent.parent.name
        if name in seen: raise ValueError(f"Multiple primary configurations for {name}; select a canonical run explicitly")
        seen.add(name)
        records.extend(rows)
    if not records: raise ValueError("No completed examples to analyze")
    out = ROOT/("results/report-preview" if preview else "results/report")
    out.mkdir(parents=True,exist_ok=True)
    flat = [{k:v for k,v in r.items() if k not in ("metrics","probabilities")} | r["metrics"] for r in records]
    df = pd.DataFrame(flat)
    df.to_parquet(out/"per_example.parquet",index=False)
    coverage = df.groupby(["model","dataset"]).size().unstack(fill_value=0)
    coverage.to_csv(out/"coverage.csv")
    complete = all(name in coverage.index and all(coverage.loc[name].get(d,0)==n for d,n in [("trec",500),("clinc",1000),("massive",1000)])
                   for name in CATEGORICAL_MODELS)
    from .audit_report import audit_report
    audit_report()
    audit_runs = read_json(ROOT/'results/report/audit_coverage.json')['runs']
    def has_audit(model,permutation=0,repeat=0,precision=None,n=90):
        return any(r['model']==model and r['permutation']==permutation and r['repeat']==repeat
                   and (precision is None or r['precision']==precision) and r['examples']==n for r in audit_runs)
    audits_complete = (all(has_audit(model,permutation=p) for model in CATEGORICAL_MODELS for p in range(1,5))
                       and all(has_audit('jev',repeat=r) for r in [1,2])
                       and all(has_audit(model,precision='fp32',n=30) for model in PRECISION_AUDIT_MODELS))
    save_json(out/"status.json",{"complete":complete and audits_complete and not preview and not PENDING_MODELS,
              "main_complete":complete,"audits_complete":audits_complete,"preview":preview,
              "model_example_records":len(df),"expected":2500*len(CATEGORICAL_MODELS),
              "active_categorical_models":CATEGORICAL_MODELS,"excluded_models":EXCLUDED_MODELS,
              "pending_protocol_models":PENDING_MODELS})
    perf,coherence,paired,subgroups = [],[],[],[]
    for (model,dataset),g in df.groupby(["model","dataset"]):
        g = g.sort_values("id")
        y = g["gold"].to_numpy()
        ix = bootstrap_indices(y,reps=100 if preview else 10000)
        values_by_structure, boot_by_structure = {}, {}
        for structure in ["coarse","flat","marginal","hard"]:
            p = np.array(g["marginal" if structure=="hard" else structure].tolist())
            truth = g["gold_parent_index"].to_numpy() if structure=="coarse" else y
            pred = g[structure+"_pred"].to_numpy()
            vals = performance(truth,p,pred)
            if structure=="hard": vals = {k:v for k,v in vals.items() if k in ("accuracy","macro_f1")}
            values_by_structure[structure] = vals
            boot_by_structure[structure] = {}
            for metric,value in vals.items():
                # Accuracy/Brier bootstrap vectorize; F1/ECE recomputed per resample.
                if metric=="accuracy": boot = (pred==truth)[ix].mean(axis=1)
                elif metric=="brier": boot = np.sum((p-np.eye(p.shape[1])[truth])**2,axis=1)[ix].mean(axis=1)
                elif metric=="ece": boot = np.array([ece(truth[j],p[j]) for j in ix])
                else:
                    boot = np.array([fast_macro_f1(truth[j],pred[j],p.shape[1]) for j in ix])
                low,high = np.quantile(boot,[.025,.975])
                boot_by_structure[structure][metric] = boot
                perf.append({"model":model,"dataset":dataset,"structure":structure,"metric":metric,
                             "value":value,"ci_low":low,"ci_high":high,"n":len(g)})
        for metric in ["cfce","coarse_tv","drift","jsd","agreement_hard","agreement_marginal"]:
            v = g[metric].to_numpy()
            boot = v[ix].mean(axis=1)
            low,high = np.quantile(boot,[.025,.975])
            coherence.append({"model":model,"dataset":dataset,"metric":metric,"mean":v.mean(),
                              "median":np.median(v),"std":v.std(ddof=1),"ci_low":low,"ci_high":high,"n":len(v)})
        for structure in ["hard","marginal"]:
            for metric,value in values_by_structure[structure].items():
                delta = value-values_by_structure["flat"][metric]
                boot_delta = boot_by_structure[structure][metric]-boot_by_structure["flat"][metric]
                lo,hi = np.quantile(boot_delta,[.025,.975])
                paired.append({"model":model,"dataset":dataset,"comparison":structure+" minus flat", "metric":metric,
                               "difference":delta,"ci_low":lo,"ci_high":hi,"n":len(g)})
        for field in ["flat_correct","high_confidence"]:
            for value,sub in g.groupby(field):
                subgroups.append({"model":model,"dataset":dataset,"group":field,"value":bool(value),"n":len(sub),
                                  **{metric:sub[metric].mean() for metric in ["cfce","drift","jsd"]}})
    tax = read_json(ROOT/"data/taxonomies.json")
    examples = read_json(ROOT/"data/examples.json")
    stats = pd.DataFrame([{"dataset":d,"parents":len(t["parents"]),"children":len(t["children"]),
                           "examples":sum(e["dataset"]==d for e in examples)} for d,t in tax.items()])
    performance_frame = pd.DataFrame(perf)
    coherence_frame = pd.DataFrame(coherence)
    paired_frame = pd.DataFrame(paired)
    performance_frame.to_csv(out/'performance_statistics.csv',index=False)
    coherence_frame.to_csv(out/'coherence_statistics.csv',index=False)
    paired_frame.to_csv(out/'paired_statistics.csv',index=False)
    def wide(frame,index,columns,value):
        result = frame.pivot(index=index,columns=columns,values=value)
        result.columns = [' '.join(c) if isinstance(c,tuple) else c for c in result.columns]
        return result.reset_index()
    tables = {"table1_datasets":stats,
              "table2_classification":wide(performance_frame[performance_frame.metric.isin(['accuracy','macro_f1'])],['dataset','model'],['structure','metric'],'value'),
              "table3_calibration":wide(performance_frame[performance_frame.metric.isin(['brier','ece'])],['dataset','model'],['structure','metric'],'value'),
              "table4_coherence":wide(coherence_frame,['dataset','model'],'metric','mean'),
              "table5_paired":wide(paired_frame,['dataset','model','comparison'],'metric','difference')}
    for name,table in tables.items():
        table.to_csv(out/(name+".csv"),index=False)
    pd.DataFrame(subgroups).to_csv(out/"subgroups.csv",index=False)
    for metric,filename,title in [("cfce","figure2_cfce","Coarse–fine consistency error"),("drift","figure3_drift","Gold-label probability drift")]:
        fig,axes = plt.subplots(1,3,figsize=(12,3.8))
        if not complete or preview: fig.suptitle("PARTIAL / VALIDATION PREVIEW — not final study results",fontsize=10)
        for ax,d in zip(axes,["trec","clinc","massive"]):
            for model,g in df[df.dataset==d].groupby("model"):
                v = np.sort(g[metric].to_numpy()); ax.plot(v,np.arange(1,len(v)+1)/len(v),label=model)
            ax.set(title=d.upper(),xlabel=title,ylabel="Cumulative fraction")
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ["pdf","png"]: fig.savefig(out/(filename+"."+ext),dpi=250)
        plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,4)); ax.axis("off")
    boxes = [("Input",.5,.9),("Direct coarse\nP(parent)",.15,.55),("Flat fine\nP(child)",.5,.55),
             ("Conditional children\nAll parents",.85,.55),("Hard routing + marginalization\nAccuracy, calibration, coherence",.5,.12)]
    for label,x,y in boxes: ax.text(x,y,label,ha="center",va="center",bbox=dict(boxstyle="round,pad=.5",fc="#edf3fb",ec="#42618a"))
    for x in [.15,.5,.85]:
        ax.annotate("",xy=(x,.68),xytext=(.5,.82),arrowprops=dict(arrowstyle="->"))
        ax.annotate("",xy=(.5,.25),xytext=(x,.43),arrowprops=dict(arrowstyle="->"))
    for ext in ["pdf","png"]: fig.savefig(out/("figure1_framework."+ext),dpi=250,bbox_inches="tight")
    plt.close(fig)
    print("Analysis complete; full coverage:",complete,flush=True)


def fast_macro_f1(y,pred,k):
    cm = np.bincount(y*k+pred,minlength=k*k).reshape(k,k)
    denom = cm.sum(0)+cm.sum(1)
    return np.divide(2*np.diag(cm),denom,out=np.zeros(k,dtype=float),where=denom>0).mean()
