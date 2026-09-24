# memopro

Goal: let any **PyTorch developer** with little memory use models that need much more memory in their own projects, services, development and training (GPU/RAM, not agent memory). Not targeted: no-code LLM app end users, and memopro does not route users to external tools (0012: S1, S3 approved; S2 rejected). Rust core (crates.io `memopro`) + PyO3/maturin Python package (PyPI `memopro`).

Two layers (0010): **universal access layer** (env detection → budget → candidate configurations → apply fail-open → report; entry points CLI/`load`/`optimize`/`train_session`/notebook/integrations) wraps existing proven techniques as optional backends — **never reimplement them**; **research core** = memopro-native techniques (α rfc, β hibernate, γ elastic, census) — novelty rule applies here only. Direction history lives in docs/research 0001–0013. Current design v0.3.1: v0.1 = doctor + census (redundancy measurement is the differentiator) + β hibernate; v0.2 = access layer (optimize/train_session/load/check); v0.3 = α (gate Gα); v0.4 = γ + `memopro run`.

Design: `docs/design/architecture.md` (two layers, principles U1–U9 + P1–P7, Technique interface, candidate configurationss) and `docs/design/development-plan.md` (tracks A/R/N, gates Gα/Gγ, Definition of Done). Follow them; if a change deviates, record a decision entry first.

Novelty rule: before building any technique, check prior art and record the check; say "not found in our survey", never "novel".

## Build & test

```bash
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release   # build Rust extension into .venv
.venv/bin/pytest -q && .venv/bin/ruff check python tests experiments && .venv/bin/ruff format --check python tests experiments
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p memopro
```
Experiments: `.venv/bin/python -m experiments.<name>.<script>`; results + `env.json` go to `docs/research/data/eNNN/`; model/dataset caches in `.cache/` (git-ignored). Never publish to PyPI/crates.io without the user's explicit confirmation.

## Research log — mandatory

The user will write research reports / papers from this project. **Every step must be documented in `docs/research/`**:

- After each meaningful step (survey, design decision, implementation milestone, experiment, benchmark, reverted/failed attempt), add a new entry `docs/research/NNNN-short-title.md` from `_template.md` and add it to the index in `docs/research/README.md`.
- Never rewrite past entries; if a decision changes, write a new entry and mark the old one `대체됨(→ NNNN)`.
- Experiments must record environment, exact commands, inputs, and where raw results are stored (`docs/research/data/`).
- Add citations to `docs/research/references.bib`; never invent bibliographic fields — mark unknowns with `note = {TODO: ...}`.
- End every entry with a "논문 매핑" section.
- Also update README.md roadmap/status when a phase changes.

Documents are written in Korean.

## Dev machine

MacBook Air M1, 8GB unified memory (itself a low-memory testbed). CUDA validation needs an external GPU environment.
