# System One Coherence

Experiment code and recorded results for **Do System One Decisions Add Up? A Study of Probabilistic Coherence**, by Saman Sarker Joy, Faculty of Computer Science and Information Technology, Universiti Malaya, Kuala Lumpur, Malaysia.

This repository examines whether decision probabilities agree when the same classification problem is asked directly or broken into broad categories and finer choices. It contains the completed Jev and English Laya evaluation: question construction, native responses, normalized predictions, statistical analysis, robustness audits, and scientific plots.

## What is being tested?

For each input, the model answers:

1. A direct question over every fine label, producing probabilities p(c).
2. A question over broad categories, producing probabilities u(k).
3. A fine-label question within **every** category, producing probabilities v(c | k).

The reconstructed distribution is q(c) = u(parent(c)) × v(c | parent(c)).

Category coherence compares u(k) with the sum of direct probabilities of that category's children. Reconstruction coherence compares p(c) with q(c). Accuracy, calibration, and choice-order sensitivity describe additional aspects of decision quality.

## Evaluation and results

| Dataset | Examples per system | Broad categories | Fine labels |
| --- | ---: | ---: | ---: |
| TREC | 500 | 6 | 50 |
| CLINC150 | 1,000 | 10 | 150 |
| MASSIVE, English | 1,000 | 18 | 60 |

The main evaluation contains **5,000 model-example records and 72,000 classification questions**. There are **900 additional audit records**: four option-order permutations on 90 examples per system, plus two repeated-request evaluations on 90 Jev examples.

| Dataset | Jev direct accuracy | Jev reconstructed accuracy | Laya direct accuracy | Laya reconstructed accuracy |
| --- | ---: | ---: | ---: | ---: |
| TREC | 72.2% | 60.4% | 28.8% | 34.4% |
| CLINC150 | 91.0% | 68.1% | 23.1% | 44.4% |
| MASSIVE | 83.3% | 66.9% | 23.8% | 33.0% |

On CLINC150, reconstruction changes accuracy by -22.9 percentage points for Jev (paired 95% bootstrap interval [-24.9, -20.9]) and +21.3 points for Laya ([18.0, 24.5]). Laya's MASSIVE accuracy improves while its expected calibration error increases from 0.046 to 0.124.

These results describe the recorded systems, samples, and question formulations. Instructions, candidate sets, and released calibration change jointly. Coherence concerns compatibility across those formulations. Intervals use matched examples, 10,000 resamples within fine-label strata, and no multiple-comparison adjustment.

## Repository contents

| Location | Contents |
| --- | --- |
| coherence/ | Data preparation, questions, model interfaces, analysis, audits, validation |
| tests/ | Probability, analysis, resumability, and sensitivity tests |
| data/ | Frozen examples, taxonomies, audit IDs, and source manifest |
| results/jev/ and results/laya/ | Run manifests and all 5,900 normalized records |
| results/cache-archives/ | Compressed native question/response caches |
| results/report/ | CSV statistics, Parquet records, coverage, audit summaries |
| results/figures/ | Four empirical plots and supporting diagnostics |
| scripts/restore_caches.py | Offline restoration of raw question caches |
| sources.lock.json | Recorded dataset and model revisions |
| uv.lock | Original dependency lock |
| RELEASE_NOTES.md | Packaging changes and publication checklist |

Manuscript text, manuscript LaTeX, bibliography, explanatory illustrations, weights, credentials, and local environments are excluded. Modules named paper_assets, paper_checks, and paper_review contain scientific plot/diagnostic code; their output goes to results/figures/.

## Quick start: existing results

Use Python **3.12**. Run commands from the repository root.

~~~bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux, instead:
# source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
python -m coherence.validate
python -m coherence.audit_report
python -m coherence.paper_assets
python -m coherence.paper_review
~~~

These commands use saved records and make no model requests. The supplied tables and plots can also be opened directly. Tests use synthetic responses.

To rebuild the full 10,000-resample statistical analysis:

~~~bash
python -m coherence.cli analyze
python -m coherence.paper_assets
python -m coherence.paper_review
~~~

Full analysis can take substantial time and memory. For a faster exploratory check:

~~~bash
python -m coherence.cli analyze --preview
~~~

Preview output goes to results/report-preview/, separately from the completed full analysis.

## Actual inputs and native outputs

Restore the compressed caches locally:

~~~bash
python scripts/restore_caches.py
python -m coherence.paper_checks
~~~

Each raw JSON includes the example ID, exact task, native response, normalized probabilities, original probability sum, and elapsed time. Restored files appear under results/MODEL/RUN/raw/. Restoration skips identical existing files and refuses to overwrite differing files.

The cache check verifies all **72,000 main questions** against native probabilities and derived records. The archives also contain audit questions. Extracted raw directories are Git-ignored; their ZIP archives remain tracked.

To check archive integrity, release counts, and common credential patterns without extracting:

~~~bash
python scripts/verify_release.py
~~~

## New model evaluations

Existing-result analysis requires neither keys nor GPU access. New Jev requests require a TypeSafe account and can incur charges. Laya inference requires an NVIDIA CUDA GPU and downloads the pinned English checkpoint.

For the original locked dependency environment, install uv separately, then:

~~~bash
uv sync --frozen --extra local
~~~

The lock preserves the original CUDA-oriented environment; CPU-only analysis can use the lighter pip setup above. The evaluated Laya runtime is 0.3.20. Saved manifests record package versions. Hardware, serving, or dependency changes can alter outputs and run identifiers.

Copy .env.example to .env and supply your own JEV_API_KEY. Never commit .env.

~~~bash
python -m coherence.cli run --model jev
python -m coherence.cli run --model laya
python -m coherence.cli audit --model jev
python -m coherence.cli audit --model laya
~~~

These commands perform **new inference**. Preserve a separate copy of the released results first: new runs can update results/primary_runs.json. Requests are reused only when their manifest and question identifiers match.

Jev uses the recorded jev-1.13.0 categorical interface. Laya uses the English root checkpoint in FP32, released calibration, and complete-input checks. Its backend computes input budgets and rejects truncated sequences. Every category branch is queried.

The supported collection interface accepts Jev and Laya. Original internal helper definitions and lock entries are retained where useful for provenance.

## Data and metrics

Sampling uses seed 42. TREC includes its full test split. CLINC150 uses approximately balanced fine-label quotas. MASSIVE uses capacity-aware quotas. Two repeated MASSIVE texts with distinct source IDs are retained.

All declared labels remain candidates, including labels absent from the selected sample. Macro F1 averages over the full label set. Brier score sums squared errors across classes. ECE uses 15 equal-width bins and maximum normalized probability as confidence. Jensen-Shannon divergence uses base-2 logarithms.

Jev and Laya probabilities are rounded to two and four decimal places respectively. Probability mass is checked against rounding tolerance before normalization. Direct prediction, hard routing, and reconstruction use the same saved questions.

To regenerate dataset selections:

~~~bash
python -m coherence.cli prepare-data
~~~

This downloads upstream sources and writes data/. Retain sources.lock.json to preserve revisions, and back up released data before regeneration.

## Sources and reuse

- [TREC](https://huggingface.co/datasets/CogComp/trec)
- [CLINC150](https://github.com/clinc/oos-eval)
- [MASSIVE](https://github.com/alexa/massive)
- [TypeSafe / Jev](https://typesafe.ai/)
- [Laya implementation](https://github.com/NandhaKishorM/laya)
- [Laya checkpoint](https://huggingface.co/convaiinnovations/laya)

Download sources and checksums are in data/manifest.json. Dataset examples, provider responses, and upstream software remain subject to their respective terms. Review redistribution permissions before publishing. The author has not selected a code license; this package does not grant an open-source license.

## Author

Saman Sarker Joy  
Faculty of Computer Science and Information Technology  
Universiti Malaya, Kuala Lumpur, Malaysia

Repository: [samanjoy2/system-one-coherence](https://github.com/samanjoy2/system-one-coherence).
