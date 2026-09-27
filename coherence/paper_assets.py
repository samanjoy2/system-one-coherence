"""Generate manuscript tables, scientific figures and exploratory diagnostics from frozen records."""
import csv
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .core import ROOT, read_json, save_json
from .analysis import performance

OUT=ROOT/"results/figures"
MODELS=("jev","laya")
DATASETS=("trec","clinc","massive")
NAMES={"trec":"TREC","clinc":"CLINC150","massive":"MASSIVE"}
RUNS={"jev":"ca3fd3d5f7377071","laya":"a6f8115b3cb88dc5"}
COLORS={"jev":"#19677E","laya":"#C15B36"}
def records(name):
    return sorted([read_json(p) for p in (ROOT/"results"/name/RUNS[name]/"records").glob("*.json")],key=lambda r:r["id"])
def statfile(name):
    with (ROOT/"results/report"/name).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))
def tex(s):
    return str(s).replace("\\",r"\textbackslash{}").replace("&",r"\&").replace("%",r"\%").replace("_",r"\_").replace("#",r"\#")
def table(name,caption,label,cols,headers,rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/(name+".csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
def figure(name,fig):
    fig.savefig(OUT/(name+".pdf"),bbox_inches="tight")
    fig.savefig(OUT/(name+".png"),dpi=180,bbox_inches="tight")
    plt.close(fig)
def ci(row,scale=1,digits=3):
    return f"{float(row['value'])*scale:.{digits}f} [{float(row['ci_low'])*scale:.{digits}f}, {float(row['ci_high'])*scale:.{digits}f}]"

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status=read_json(ROOT/"results/report/status.json")
    assert status["complete"] and set(status["active_categorical_models"])==set(MODELS)
    data={m:records(m) for m in MODELS}
    tax=read_json(ROOT/"data/taxonomies.json")
    perf={(r["model"],r["dataset"],r["structure"],r["metric"]):r for r in statfile("performance_statistics.csv")}
    coh={(r["model"],r["dataset"],r["metric"]):r for r in statfile("coherence_statistics.csv")}
    paired={(r["model"],r["dataset"],r["comparison"],r["metric"]):r for r in statfile("paired_statistics.csv")}
    audit=statfile("audit_summary.csv")
    diagnostics=[]
    examples=[]
    confusion=[]
    for m in MODELS:
        for d in DATASETS:
            rows=[r for r in data[m] if r["dataset"]==d]
            ms=[r["metrics"] for r in rows]
            y=np.array([r["gold"] for r in ms])
            p=np.array([r["flat"] for r in ms])
            q=np.array([r["marginal"] for r in ms])
            f=p.argmax(1); h=np.array([r["hard_pred"] for r in ms]); z=q.argmax(1)
            pc=np.array([r["coarse_pred"]==r["gold_parent_index"] for r in ms])
            fc=f==y; qc=z==y
            for structure,dist in [("flat",p),("marginal",q)]:
                measured=performance(y,dist)
                for metric,value in measured.items():
                    assert np.isclose(value,float(perf[m,d,structure,metric]["value"]),atol=1e-12)
            confidence=p.max(1); high=confidence>=.8
            both=int((fc&qc).sum()); lost=int((fc&~qc).sum()); gained=int((~fc&qc).sum()); neither=int((~fc&~qc).sum())
            assert both+lost+gained+neither==len(rows)
            assert gained-lost==int(qc.sum())-int(fc.sum())
            entry={"model":m,"dataset":d,"n":len(rows),"both_correct":both,"lost":lost,"gained":gained,"both_wrong":neither,
                   "wrong_parent":int((~pc).sum()),"correct_parent_wrong_child":int((pc&(h!=y)).sum()),"hard_correct":int((h==y).sum()),
                   "marginal_rescues_hard":int(((h!=y)&qc).sum()),"marginal_loses_hard":int(((h==y)&~qc).sum()),
                   "conditional_accuracy_given_correct_parent":float((h[pc]==y[pc]).mean()),
                   "high_confidence_n":int(high.sum()),"high_confidence_errors":int((high&~fc).sum()),
                   "high_confidence_accuracy":float(fc[high].mean()) if high.any() else None,
                   "mean_signed_gold_shift":float((q[np.arange(len(y)),y]-p[np.arange(len(y)),y]).mean()),
                   "jsd_flat_correct":float(np.mean([v["jsd"] for v in ms if v["flat_correct"]])),
                   "jsd_flat_wrong":float(np.mean([v["jsd"] for v in ms if not v["flat_correct"]]))}
            assert entry["wrong_parent"]+entry["correct_parent_wrong_child"]+entry["hard_correct"]==len(rows)
            diagnostics.append(entry)
            # Post-hoc deterministic selection for illustrations, not representative cases.
            pool=[r for r in rows if (r["metrics"]["flat_pred"]==r["metrics"]["gold"]) != (r["metrics"]["marginal_pred"]==r["metrics"]["gold"])]
            pool=[r for r in pool if (r["metrics"]["flat_pred"]==r["metrics"]["gold"])==(m=="jev")]
            chosen=sorted(pool,key=lambda r:(-r["metrics"]["drift"],r["id"]))[0]
            metric=chosen["metrics"]; labels=sorted(tax[d]["children"]); parents=sorted(tax[d]["parents"])
            examples.append({"model":m,"dataset":d,"id":chosen["id"],"text":chosen["text"],"gold":chosen["gold_child"],
                             "flat":labels[metric["flat_pred"]],"marginal":labels[metric["marginal_pred"]],
                             "gold_parent":chosen["gold_parent"],"coarse":parents[metric["coarse_pred"]],
                             "p_gold":metric["flat"][metric["gold"]],"q_gold":metric["marginal"][metric["gold"]],
                             "coarse_gold":metric["coarse"][metric["gold_parent_index"]]})
            labels=sorted(tax[d]["children"])
            for structure,pred in [("flat",f),("marginal",z)]:
                counts=Counter((labels[a],labels[b]) for a,b in zip(y,pred) if a!=b)
                confusion.extend({"model":m,"dataset":d,"structure":structure,"gold":a,"predicted":b,"count":n}
                                 for (a,b),n in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))[:5])
    save_json(OUT/"diagnostics.json",{"groups":diagnostics,"illustrative_cases":examples,"top_confusions":confusion,
                                   "case_selection":"Per model/dataset: greatest absolute gold-probability drift among Jev lost-correct or Laya gained-correct cases; ties by ID."})
    rows=[]
    for d in DATASETS:
        for m in MODELS:
            vals=[NAMES[d],m.title()]
            for s in ("flat","hard","marginal"):
                vals += [f"{float(perf[m,d,s,'accuracy']['value'])*100:.1f}",f"{float(perf[m,d,s,'macro_f1']['value'])*100:.1f}"]
            rows.append(vals)
    table("classification","Fine-label performance (percent). Macro F1 includes every declared class, including absent test classes. F, H, and M denote flat, hard routing, and marginalization.","tab:classification","llrrrrrr",
          ["Dataset","Model","F acc.","F F1","H acc.","H F1","M acc.","M F1"],rows)
    rows=[]
    for d in DATASETS:
        for m in MODELS:
            r=paired[m,d,"marginal minus flat","accuracy"]
            rows.append([NAMES[d],m.title(),ci(perf[m,d,"flat","accuracy"],100,1),ci(perf[m,d,"marginal","accuracy"],100,1),
                         f"{float(r['difference'])*100:+.1f} [{float(r['ci_low'])*100:+.1f}, {float(r['ci_high'])*100:+.1f}]"])
    table("accuracy_ci","Accuracy and paired changes with 95\\% stratified percentile-bootstrap intervals (10,000 replicates). Accuracies are percentages; changes are percentage points.","tab:accuracyci","llrrr",
          ["Dataset","Model","Flat","Marginal","Change"],rows)
    rows=[]
    for d in DATASETS:
        for m in MODELS:
            rows.append([NAMES[d],m.title()]+[f"{float(perf[m,d,s,k]['value']):.3f}" for s in ("coarse","flat","marginal") for k in ("brier","ece")])
    table("calibration","Brier score (B) and 15-bin top-label ECE (E), lower is better. Brier uses the sum of classwise squared errors. Coarse scores concern a different label space.","tab:calibration","llrrrrrr",
          ["Dataset","Model","Coarse B","Coarse E","Flat B","Flat E","Marg. B","Marg. E"],rows)
    rows=[]
    for d in DATASETS:
        for m in MODELS:
            rows.append([NAMES[d],m.title()]+[f"{float(coh[m,d,k]['mean']):.3f}" for k in ("cfce","coarse_tv","drift","jsd")]+[f"{100*float(coh[m,d,'agreement_marginal']['mean']):.1f}"])
    table("coherence","Mean coherence discrepancies and flat--marginal label agreement. TV, drift, and JSD range from zero to one; JSD uses bits. CFCE and TV are algebraically redundant; compare TV across datasets.","tab:coherence","llrrrrr",
          ["Dataset","Model","CFCE","Parent TV","Gold drift","JSD","Agree (\\%)"],rows)
    rows=[]
    for r in diagnostics:
        rows.append([NAMES[r["dataset"]],r["model"].title(),r["both_correct"],r["lost"],r["gained"],r["both_wrong"],r["wrong_parent"],r["correct_parent_wrong_child"]])
    table("transitions","Paired error accounting (counts). Both/lost/gained/neither refer to correctness under flat and marginal predictions and sum to N. Wrong parent plus within-parent errors plus hard-correct cases separately sum to N.","tab:errors","llrrrrrr",
          ["Dataset","Model","Both","Lost","Gained","Neither","Wrong parent","Within"],rows)
    rows=[]
    for d in DATASETS:
        for m in MODELS:
            r=[v for v in audit if v["model"]==m and v["dataset"]==d and v["audit"]=="order"]
            by={v["structure"]:v for v in r}
            rows.append([NAMES[d],m.title()]+[f"{float(by[s][k])*100:.1f}" if k=='prediction_flip' else f"{float(by[s][k]):.3f}" for s in ("flat","marginal") for k in ("tv","prediction_flip")])
    table("order","Option-order audit: mean TV and label-flip percentage against the main order. Each row uses 30 examples with four permutations (120 paired comparisons sharing 30 inputs).","tab:order","llrrrr",
          ["Dataset","Model","Flat TV","Flat flip (\\%)","Marg. TV","Marg. flip (\\%)"],rows)
    rows=[]
    for d in DATASETS:
        by={v["structure"]:v for v in audit if v["model"]=="jev" and v["dataset"]==d and v["audit"]=="repeat"}
        rows.append([NAMES[d]]+[f"{float(by[s][k])*100:.1f}" if k=='prediction_flip' else f"{float(by[s][k]):.3f}" for s in ("flat","marginal") for k in ("tv","prediction_flip")])
    table("repeat","Jev identical-request audit: 30 examples per dataset, each repeated twice (60 paired comparisons). These are descriptive summaries of rounded outputs.","tab:repeat","lrrrr",
          ["Dataset","Flat TV","Flat flip (\\%)","Marg. TV","Marg. flip (\\%)"],rows)
    rows=[]
    for r in diagnostics:
        rows.append([NAMES[r["dataset"]],r["model"].title(),r["high_confidence_n"],r["high_confidence_errors"],f"{100*r['high_confidence_accuracy']:.1f}",
                     f"{r['jsd_flat_correct']:.3f}",f"{r['jsd_flat_wrong']:.3f}"])
    table("subgroups","Exploratory flat-confidence and correctness subgroups. High confidence means maximum flat probability at least 0.8. JSD columns condition on flat correctness; subgroup membership is not randomized.","tab:subgroups","llrrrrr",
          ["Dataset","Model","High-conf. N","Errors","Acc. (\\%)","JSD correct","JSD wrong"],rows)
    rows=[]
    for d in DATASETS:
        for m in MODELS:
            r=coh[m,d,"coarse_tv"]; j=coh[m,d,"jsd"]
            rows.append([NAMES[d],m.title()]+[f"{float(v[k]):.3f}" for v in (r,j) for k in ("mean","ci_low","ci_high")])
    table("coherence_ci","Mean parent TV and JSD with 95\\% stratified bootstrap intervals.","tab:coherenceci","llrrrrrr",
          ["Dataset","Model","TV","Low","High","JSD","Low","High"],rows)
    rows=[]
    for d in DATASETS:
        for m in MODELS:
            row=[NAMES[d],m.title()]
            for metric in ('brier','ece'):
                r=paired[m,d,'marginal minus flat',metric]
                row.append(f"{float(r['difference']):+.3f} [{float(r['ci_low']):+.3f}, {float(r['ci_high']):+.3f}]")
            rows.append(row)
    table('calibration_ci','Paired marginal-minus-flat changes in Brier and ECE with 95\\% bootstrap intervals. Negative changes indicate lower loss or calibration error.','tab:calibrationci','llrr',
          ['Dataset','Model','Brier change','ECE change'],rows)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.4),sharey=True,layout="constrained")
    for ax,d in zip(axes,DATASETS):
        for j,m in enumerate(MODELS):
            for k,s in enumerate(("flat","hard","marginal")):
                r=perf[m,d,s,"accuracy"]; value=float(r["value"])*100
                ax.errorbar(k+(j-.5)*.13,value,yerr=[[value-float(r["ci_low"])*100],[float(r["ci_high"])*100-value]],fmt="o" if m=="jev" else "s",capsize=3,color=COLORS[m],label=m.title() if k==0 else None)
        ax.set(title=NAMES[d],xticks=[0,1,2],xticklabels=["Flat","Hard","Marginal"],ylim=(0,100));ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Accuracy (%)");axes[-1].legend(loc="upper right")
    figure("accuracy",fig)
    fig,axes=plt.subplots(2,3,figsize=(10.5,5.0),layout="constrained")
    for c,d in enumerate(DATASETS):
        for row,metric in enumerate(("coarse_tv","jsd")):
            ax=axes[row,c]
            for m in MODELS:
                v=np.sort([r["metrics"][metric] for r in data[m] if r["dataset"]==d])
                ax.step(np.r_[0,v,1],np.r_[0,np.arange(1,len(v)+1)/len(v),1],where="post",label=m.title(),color=COLORS[m],linestyle="-" if m=="jev" else "--")
            ax.set(xlim=(0,1),ylim=(0,1),xlabel="Parent total variation" if row==0 else "Jensen-Shannon divergence (bits)")
            if row==0:ax.set_title(NAMES[d])
            if c==0:ax.set_ylabel("Cumulative fraction")
            ax.grid(alpha=.2)
    axes[0,-1].legend(loc="lower right");figure("coherence_ecdf",fig)
    fig,axes=plt.subplots(2,3,figsize=(10.5,5.2),sharex=True,sharey=True,layout="constrained")
    reliability=[]
    for row,m in enumerate(MODELS):
        for c,d in enumerate(DATASETS):
            ax=axes[row,c];ax.plot([0,1],[0,1],":",color="0.6")
            ms=[r["metrics"] for r in data[m] if r["dataset"]==d]
            for s,color in [("flat","#19677E"),("marginal","#C15B36")]:
                p=np.array([v[s] for v in ms]);y=np.array([v["gold"] for v in ms]);conf=p.max(1);correct=p.argmax(1)==y;bins=np.minimum((conf*15).astype(int),14)
                points=[]
                for b in range(15):
                    mask=bins==b
                    if mask.any():
                        x=float(conf[mask].mean());z=float(correct[mask].mean());n=int(mask.sum());points.append((x,z,n))
                        reliability.append({"model":m,"dataset":d,"structure":s,"bin":b,"n":n,"confidence":x,"accuracy":z})
                arr=np.array(points)
                ax.plot(arr[:,0],arr[:,1],color=color,alpha=.8,label=s.title(),linestyle="-" if s=="flat" else "--")
                ax.scatter(arr[:,0],arr[:,1],s=10+60*arr[:,2]/len(y),color=color,marker="o" if s=="flat" else "s")
            ax.set(title=m.title()+" / "+NAMES[d],xlim=(0,1),ylim=(0,1));ax.grid(alpha=.15)
            if row==1:ax.set_xlabel("Mean confidence")
            if c==0:ax.set_ylabel("Observed accuracy")
    axes[0,-1].legend(loc="upper left");figure("reliability",fig)
    save_json(OUT/"reliability_bins.json",reliability)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),layout="constrained")
    for ax,m in zip(axes,MODELS):
        rr=[next(r for r in diagnostics if r["model"]==m and r["dataset"]==d) for d in DATASETS]
        left=np.zeros(3)
        for key,label,color in [("both_correct","Both correct","#19677E"),("lost","Lost correctness","#C15B36"),("gained","Gained correctness","#5E9C7B"),("both_wrong","Both wrong","#CDD3D8")]:
            vals=np.array([100*r[key]/r["n"] for r in rr])
            ax.barh(range(3),vals,left=left,label=label,color=color);left+=vals
        ax.set(title=m.title(),yticks=range(3),yticklabels=[NAMES[d] for d in DATASETS],xlim=(0,100),xlabel="Examples (%)");ax.invert_yaxis()
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=4,frameon=False);figure("error_transitions",fig)
    print("Generated paper assets and diagnostics",flush=True)
    print(diagnostics,flush=True)
    print(examples,flush=True)
if __name__=="__main__":main()
