# experiments

Research experiments for memopro. Each experiment lives in `eNNN_<name>/` and writes curated
results to `docs/research/data/eNNN/` together with `env.json` (see `_harness/env.py`).

Rules (CLAUDE.md, docs/research/README.md):
- Acceptance criteria are pre-registered in a research log entry **before** running.
- Every run records environment, exact command, inputs and the script SHA-256.
- Model and dataset downloads go to the project-local `.cache/` (git-ignored).

Run from the repository root with the project virtual environment, e.g.

```bash
.venv/bin/python -m experiments.e001_e003_rfc.run --help
```
