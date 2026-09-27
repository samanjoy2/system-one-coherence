from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def distribution(raw, labels, decimals=4):
    if set(raw) != set(labels):
        raise ValueError("Probability labels do not match candidate set")
    p = np.array([raw[k] for k in labels], dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid probabilities")
    total = float(p.sum())
    # Provider-specific rounding bound; local model vectors use the strict default.
    if total <= 0 or abs(total - 1) > len(labels) * (0.5*10**(-decimals)) + 1e-6:
        raise ValueError(f"Probability mass is {total}")
    return p / total, total


def ordered(labels, example_id, permutation=0):
    seed = int(digest([SEED, example_id, permutation])[:16], 16)
    return np.random.default_rng(seed).permutation(sorted(labels)).tolist()


def questions(example, taxonomy, permutation=0):
    parents, children = taxonomy["parents"], taxonomy["children"]
    tasks = [("coarse", parents, "Classify the input into one of these broad categories.")]
    tasks.append(("flat", children, "Classify the input into one of these specific categories."))
    for parent, desc in parents.items():
        subset = {c: v for c, v in children.items() if taxonomy["child_parent"][c] == parent}
        tasks.append((f"conditional:{parent}", subset,
                      f"Assuming the input belongs to {desc}, classify it into one of these specific categories."))
    output = []
    for name, labels, instruction in tasks:
        order = ordered(labels, example["id"], permutation)
        # Stable semantic IDs, independent of display position and condition.
        universe = sorted(parents if name == "coarse" else children)
        ids = {label: f"L{universe.index(label):03d}" for label in order}
        output.append({"name": name, "state": example["text"], "instructions": instruction,
                       "criteria": {ids[k]: labels[k] for k in order},
                       "id_label": {ids[k]: k for k in order}})
    return output


def derive(example, taxonomy, probabilities):
    ps, cs = sorted(taxonomy["parents"]), sorted(taxonomy["children"])
    coarse, _ = distribution(probabilities["coarse"], ps)
    flat, _ = distribution(probabilities["flat"], cs)
    parent_index = np.array([ps.index(taxonomy["child_parent"][c]) for c in cs])
    hier = np.zeros(len(cs))
    for j, parent in enumerate(ps):
        subset = [c for c in cs if taxonomy["child_parent"][c] == parent]
        conditional, _ = distribution(probabilities[f"conditional:{parent}"], subset)
        hier[parent_index == j] = coarse[j] * conditional
    assert np.isclose(hier.sum(), 1)
    selected_parent = int(coarse.argmax())
    eligible = np.flatnonzero(parent_index == selected_parent)
    hard_idx = int(eligible[hier[eligible].argmax()])
    aggregate = np.bincount(parent_index, weights=flat, minlength=len(ps))
    midpoint = (flat + hier) / 2
    def kl(p):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / midpoint[mask])))
    gold = cs.index(example["gold_child"])
    return {"coarse": coarse.tolist(), "flat": flat.tolist(), "marginal": hier.tolist(),
            "coarse_pred": int(coarse.argmax()), "flat_pred": int(flat.argmax()),
            "marginal_pred": int(hier.argmax()), "hard_pred": hard_idx,
            "gold": gold, "gold_parent_index": ps.index(example["gold_parent"]),
            "cfce": float(np.abs(coarse - aggregate).mean()),
            "coarse_tv": float(np.abs(coarse - aggregate).sum() / 2),
            "drift": float(abs(flat[gold] - hier[gold])), "jsd": (kl(flat) + kl(hier)) / 2,
            "agreement_hard": int(flat.argmax() == hard_idx),
            "agreement_marginal": int(flat.argmax() == hier.argmax()),
            "flat_correct": bool(flat.argmax() == gold), "high_confidence": bool(flat.max() >= .8)}
