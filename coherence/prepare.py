"""Download immutable source snapshots and reproducible evaluation samples."""
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import requests

from .core import ROOT, SEED, digest, read_json, save_json


def fetch(url):
    for attempt in range(5):
        r = requests.get(url, timeout=180)
        if r.status_code in (429,500,502,503,504) and attempt < 4:
            time.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r.content


def github_pin(repo, branch):
    return json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{branch}"))["sha"]


def prepare_sources():
    path = ROOT / "sources.lock.json"
    if path.exists():
        return read_json(path)
    sources = {"github": {}, "models": {}}
    for repo, branch in [("clinc/oos-eval", "master"), ("jaredpalmer/kev", "main"),
                         ("NandhaKishorM/laya", "main")]:
        sources["github"][repo] = github_pin(repo, branch)
    for repo in ["Qwen/Qwen3.5-4B-Base", "jaredpalmer/kev-4b", "convaiinnovations/laya",
                 "Qwen/Qwen3.5-0.8B-Base"]:
        # Last entry is metadata only, never an evaluation model.
        if "0.8B" in repo:
            continue
        sources["models"][repo] = json.loads(fetch(f"https://huggingface.co/api/models/{repo}"))["sha"]
    sources["trec_revision"] = json.loads(fetch("https://huggingface.co/api/datasets/CogComp/trec"))["sha"]
    save_json(path, sources)
    return sources


def get_raw(name, url):
    path = ROOT / "data/raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(fetch(url))
    import hashlib
    return path.read_bytes(), {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def stratified(rows, n, classes, seed=SEED):
    rng = np.random.default_rng(seed)
    labels = sorted(classes)
    quotas = {k: n // len(labels) for k in labels}
    for k in rng.permutation(labels)[:n % len(labels)]:
        quotas[k] += 1
    capacity = {k:sum(r["gold_child"] == k for r in rows) for k in labels}
    quotas = {k:min(quotas[k],capacity[k]) for k in labels}
    # MASSIVE has rare test intents: preserve all examples in those strata and
    # redistribute their shortfall across the least represented eligible strata.
    tie_order = rng.permutation(labels).tolist()
    while sum(quotas.values()) < n:
        eligible = [k for k in tie_order if quotas[k] < capacity[k]]
        if not eligible: raise ValueError("Requested sample exceeds test population")
        k = min(eligible,key=lambda x:quotas[x])
        quotas[k] += 1
    selected = []
    for label in labels:
        candidates = sorted([r for r in rows if r["gold_child"] == label], key=lambda r:r["id"])
        if len(candidates) < quotas[label]:
            raise ValueError(f"Insufficient test examples for {label}: {len(candidates)} < {quotas[label]}")
        selected.extend(candidates[i] for i in rng.choice(len(candidates), quotas[label], replace=False))
    return sorted(selected, key=lambda r:r["id"])


def prepare_data():
    import pandas as pd
    lock = prepare_sources()
    manifest, all_rows, taxonomies = {}, [], {}
    rev = lock["github"]["clinc/oos-eval"]
    base = f"https://raw.githubusercontent.com/clinc/oos-eval/{rev}"
    raw, manifest["clinc"] = get_raw("clinc.json", base + "/data/data_full.json")
    # Official repository taxonomy, not a guessed partition of intent names.
    tree = json.loads(fetch(f"https://api.github.com/repos/clinc/oos-eval/git/trees/{rev}?recursive=1"))
    candidates = [p["path"] for p in tree["tree"] if "domain" in p["path"].lower() and p["path"].endswith(".json")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one official domain mapping, found {candidates}")
    mapping_raw, manifest["clinc_taxonomy"] = get_raw("clinc_domains.json", base + "/" + candidates[0])
    mapping = json.loads(mapping_raw)
    assert len(mapping) == 10 and all(len(v) == 15 for v in mapping.values())
    child_parent = {c: p for p, cs in mapping.items() for c in cs}
    assert len(child_parent) == 150
    taxonomies["clinc"] = {"parents": {p:p.replace("_", " ") for p in mapping},
                           "children": {c:c.replace("_", " ") for c in child_parent}, "child_parent": child_parent}
    rows = [{"dataset":"clinc", "id":f"clinc:test:{i}", "text":text, "gold_child":label,
             "gold_parent":child_parent[label]} for i,(text,label) in enumerate(json.loads(raw)["test"])]
    all_rows.extend(stratified(rows, 1000, child_parent))
    raw, manifest["massive"] = get_raw("massive.tar.gz", "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        member = next(m for m in archive.getmembers() if m.name.endswith("/en-US.jsonl"))
        english = [json.loads(line) for line in archive.extractfile(member)]
    child_parent = {}
    for r in english:
        assert child_parent.setdefault(r["intent"],r["scenario"]) == r["scenario"]
    assert len(child_parent) == 60 and len(set(child_parent.values())) == 18
    taxonomies["massive"] = {"parents":{p:p.replace("_"," ") for p in set(child_parent.values())},
                              "children":{c:c.replace("_"," ") for c in child_parent}, "child_parent":child_parent}
    rows = [{"dataset":"massive", "id":f"massive:test:{r['id']}", "text":r["utt"],
             "gold_child":r["intent"], "gold_parent":r["scenario"]} for r in english if r["partition"] == "test"]
    all_rows.extend(stratified(rows,1000,child_parent))
    rev = lock["trec_revision"]
    if "trec_parquet_revision" not in lock:
        lock["trec_parquet_revision"] = json.loads(fetch("https://huggingface.co/api/datasets/CogComp/trec/revision/refs%2Fconvert%2Fparquet"))["sha"]
        save_json(ROOT/"sources.lock.json",lock)
    rev = lock["trec_parquet_revision"]
    tree = json.loads(fetch(f"https://huggingface.co/api/datasets/CogComp/trec/tree/{rev}?recursive=true"))
    files = [x["path"] for x in tree if "test" in x["path"] and x["path"].endswith(".parquet")]
    if len(files) != 1:
        raise ValueError(f"Expected one TREC test parquet: {files}")
    raw, manifest["trec"] = get_raw("trec.parquet",f"https://huggingface.co/datasets/CogComp/trec/resolve/{rev}/{files[0]}")
    import pyarrow.parquet as pq
    table = pq.read_table(io.BytesIO(raw))
    metadata = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
    coarse_names = metadata["coarse_label"]["names"]
    fine_names = metadata["fine_label"]["names"]
    coarse_desc = {"ABBR":"Abbreviation", "DESC":"Description", "ENTY":"Entity", "HUM":"Human", "LOC":"Location", "NUM":"Numeric"}
    child_parent = {c:c.split(":")[0] for c in fine_names}
    taxonomies["trec"] = {"parents":{p:coarse_desc[p] for p in coarse_names},
                           "children":{c:coarse_desc[child_parent[c]] + ": " + c.split(":")[1].replace("_"," ") for c in fine_names},
                           "child_parent":child_parent}
    rows = [{"dataset":"trec", "id":f"trec:test:{i}", "text":r["text"],
             "gold_child":fine_names[r["fine_label"]], "gold_parent":coarse_names[r["coarse_label"]]}
            for i,r in enumerate(table.to_pylist())]
    assert len(rows) == 500
    all_rows.extend(rows)
    save_json(ROOT/"data/taxonomies.json",taxonomies)
    save_json(ROOT/"data/examples.json",all_rows)
    save_json(ROOT/"data/manifest.json",{"seed":SEED,"sources":manifest,"examples_hash":digest(all_rows),
              "taxonomy_hash":digest(taxonomies),"counts":{d:sum(r["dataset"]==d for r in all_rows) for d in taxonomies}})
    print("Prepared",len(all_rows),"examples",flush=True)


if __name__ == "__main__":
    prepare_data()
