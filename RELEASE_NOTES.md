# Companion repository packaging

Prepared 2026-09-28 as a separate copy of the completed Jev/Laya experiment workspace. The original workspace is unchanged.

## Evidence

Frozen benchmark selections, canonical main runs, all completed Jev/Laya order audits and Jev repeat audits, manifests, normalized records, compressed native responses, full statistics, and four empirical plots are included.

Canonical runs: Jev ca3fd3d5f7377071; Laya a6f8115b3cb88dc5.

## Packaging changes

- Supported CLI and backend entry point accept Jev and Laya only.
- Scientific plot and diagnostic outputs go to results/figures/. Generated tables use CSV; LaTeX table output is omitted.
- Primary-run selection explicitly names both canonical runs.
- Package-version collection tolerates absent optional inference dependencies.
- Manuscript text, LaTeX, bibliography, schematics, environments, weights, logs, failures, credentials, and unrelated model results are excluded.
- Original dependency/source locks are intact. Historical entries and unused internal backend definitions remain for provenance.

Tests cover core probabilities, analysis, resumability, and sensitivity diagnostics. Manuscript-specific and unrelated-runner tests are outside this package.

## Before publication

1. Choose a license for code you own.
2. Confirm permission to redistribute benchmark text and provider responses; both are included here.
3. Add the eventual paper and repository URLs.
4. Keep .env and extracted raw caches out of Git.
5. Upload this folder's contents as the repository root.

Publication target: https://github.com/samanjoy2/system-one-coherence
