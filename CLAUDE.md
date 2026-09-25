# memopro

Goal: let any **PyTorch developer** with little memory use models that need much more memory in their own projects, services, development and training (GPU/RAM, not agent memory). Not targeted: no-code LLM app end users, and memopro does not route users to external tools (0012: S1, S3 approved; S2 rejected). Rust core (crates.io `memopro`) + PyO3/maturin Python package (PyPI `memopro`).

Two layers (0010): **universal access layer** (env detection → budget → candidate configurations → apply fail-open → report; entry points CLI/`load`/`optimize`/`train_session`/notebook/integrations) wraps existing proven techniques as optional backends — **never reimplement them**; **research core** = memopro-native techniques (β hibernate, census, γ elastic; α rfc rejected in 0018) — novelty rule applies here only. Direction history lives in docs/research 0001–0027. Current design v0.3.5 / plan v0.3.4: v0.1 = doctor + census + β hibernate (explicit `%hibernate`; CPU default = uncompressed SSD spill, CUDA default = move to host, bf16 only on explicit request); v0.2 = access layer (optimize/train_session/load/check); v0.3 = γ + `memopro run`. **α (residual fixed-point checkpointing) was REJECTED by the pre-registered X1 experiments (0018)** — do not revive it without a new decision entry. Open questions for the user: 0021 Q1–Q3, 0022 R1–R5. Minimum Python is 3.11 (abi3-py311, 0026). Muon (torch.optim.Muon) was evaluated in 0024: not a train_session candidate (3x slower on M1 small batches; bf16 Newton–Schulz; worse for full fine-tuning of Adam-pretrained models). Rust lesson (0025): naive Rust lost to Python threads; engineered Rust (buffer reuse, specialised shuffle, pipeline owned end-to-end, no boundary copies) won. **Rust core rules RS1–RS5 are adopted (0027, architecture §6.1)**: RS1 spill/restore end-to-end in one Rust call, no bulk-size Python objects; RS2 buffer-protocol input, never `bytes` (bf16 via `view(torch.uint8)`); RS3 per-thread zstd context/scratch + one process-wide pool; RS4 extra memory ≤ buffers × chunk + const via fixed buffer pool + bounded channels, RSS-tested; RS5 overlap compute and I/O. RS6–RS8 are on hold — do not build SIMD kernels, background GIL-free services or `gil_used = false` without a new decision. A1 = spill engine first, then pre-register E009; the `codec_*` bytes bindings are E008 prototypes to remove. β spill default is UNcompressed (E008 C5) until E009 says otherwise.

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
