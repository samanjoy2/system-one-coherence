import importlib.metadata
import platform
import time
import traceback
import hashlib

from .core import ROOT, digest, distribution, read_json, save_json, questions, derive


def run(model_name, limit=None, permutation=0, repeat=0, precision="int8", audit=False, instance=None):
    from .backends import backend
    rows = read_json(ROOT/"data/examples.json")
    taxonomies = read_json(ROOT/"data/taxonomies.json")
    # Interleave datasets for pilots and failure discovery.
    groups = [[e for e in rows if e["dataset"] == d] for d in sorted(taxonomies)]
    rows = [g[i] for i in range(max(map(len,groups))) for g in groups if i<len(g)]
    if audit:
        from .audit import audit_examples
        rows = audit_examples(rows,taxonomies)
    if limit: rows = rows[:limit]
    model = instance if instance is not None else backend(model_name,precision)
    packages = {}
    for package in ["numpy","requests","transformers","torch","laya","bitsandbytes"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {"backend":model.meta,"packages":packages,"python":platform.python_version(),
                "data":read_json(ROOT/"data/manifest.json"),"protocol":"v1", "permutation":permutation,"repeat":repeat}
    tag = digest(manifest)[:16]
    folder = ROOT/"results"/model_name/tag
    save_json(folder/"manifest.json",manifest)
    def select_primary():
        if permutation==0 and repeat==0 and not (model_name in ('kev','qwen') and precision=='fp32'):
            selection_path = ROOT/'results/primary_runs.json'
            selection = read_json(selection_path) if selection_path.exists() else {}
            selection[model_name] = tag
            save_json(selection_path,selection)
    sources = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT/"coherence").glob("*.py"))}
    save_json(folder/"execution_sources"/(digest(sources)+".json"),sources)
    def execute(example):
        taxonomy = taxonomies[example["dataset"]]
        probs = {}
        start = time.monotonic()
        try:
            tasks = questions(example,taxonomy,permutation)
            for task in tasks:
                key = digest({"manifest":tag,"example":example,"task":task})
                target = folder/"raw"/(key+".json")
                if target.exists():
                    result = read_json(target)
                else:
                    t = time.monotonic()
                    p,raw = model.predict(task)
                    validated,total = distribution(p,list(task["criteria"]),decimals=2 if model_name=="jev" else 4)
                    result = {"example_id":example["id"],"task":task,"raw":raw,
                              "probabilities":dict(zip(task["criteria"],validated.tolist())),
                              "original_probability_sum":total,"elapsed_seconds":time.monotonic()-t}
                    save_json(target,result)
                probs[task["name"]] = {task["id_label"][k]:v for k,v in result["probabilities"].items()}
            record = example | {"model":model_name,"run":tag,"probabilities":probs,
                                  "metrics":derive(example,taxonomy,probs)}
            save_json(folder/"records"/(digest(example["id"])+".json"),record)
        except Exception as exc:
            # Provider exceptions must not include authorization headers or environment values.
            save_json(folder/"failures"/(digest(example["id"])+".json"),
                      {"example_id":example["id"],"type":type(exc).__name__,"message":str(exc)})
            raise
        return example["dataset"], time.monotonic()-start
    if model_name == "jev" or getattr(model,'workers',1)>1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2 if audit else getattr(model,'workers',8)) as pool:
            for i,(dataset,elapsed) in enumerate(pool.map(execute,rows)):
                if i==0: select_primary()
                if i%10==0 or i==len(rows)-1:
                    print(f"{model_name} {i+1}/{len(rows)} {dataset} {elapsed:.2f}s",flush=True)
    else:
        for i,example in enumerate(rows):
            dataset,elapsed = execute(example)
            if i==0: select_primary()
            if i%10==0 or i==len(rows)-1:
                print(f"{model_name} {i+1}/{len(rows)} {dataset} {elapsed:.2f}s",flush=True)
    print("Run saved:",folder,flush=True)
