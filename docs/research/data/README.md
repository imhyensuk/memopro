# Experiment data

Raw results of memopro's experiments, one directory per experiment (`eNNN/`, Colab runs `colab_run*`). Each holds the measured results (`*.json`, `summary.md`) and `env.json` (machine, software versions, commit, and the SHA-256 of the experiment scripts).

The scripts that produced them are in [`experiments/`](../../../experiments): run `.venv/bin/python -m experiments.<name>.run --all` (see each script's docstring for its cases and criteria). Numbers in the README's results tables come from these directories.

Entry numbers mentioned in scripts and data (e.g. "0198") refer to the project's research log, which is kept separately.
