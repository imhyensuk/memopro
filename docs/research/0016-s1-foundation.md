# 0016. S1 기반 구축과 걷는 뼈대

- **날짜**: 2026-09-25
- **유형**: milestone
- **상태**: 확정 — 최소 Python 3.10 → 3.11 변경(→ 0026), 이 기록의 3.10·cp310 표기는 당시 상태
- **관련 기록**: 0015 (P2 걷는 뼈대), 0008·0010 (구조 설계)

## 배경 / 동기

0015에서 채택한 P2에 따라, 기능을 만들기 전에 **빌드 → wheel → 설치 → import**와 두 레지스트리의 패키징 경로를 먼저 끝까지 관통시킨다.

## 내용

### 저장소 구성

```
Cargo.toml                    workspace (resolver 3, edition 2024, rust-version 1.85, version 0.0.1)
crates/memopro/               Rust 코어 → crates.io (현재 version()만 노출, LICENSE 동봉)
crates/memopro-py/            PyO3 0.29 바인딩, cdylib `_core`, abi3-py310, publish = false
pyproject.toml                maturin 백엔드, python-source=python, module=memopro._core, extras: torch, dev
python/memopro/               __init__ · report(Report, 실측 회수량 필드) · techniques/base(Technique 프로토콜,
                              Stage·Fidelity·Timing·Pool·Origin·QualityGrade, Registry, register_technique)
tests/test_skeleton.py        버전 일치(Cargo ↔ Python ↔ Rust), 레지스트리, 보고서
experiments/_harness/env.py   환경 자동 기록(하드웨어·OS·Python·패키지·torch·git·스크립트 SHA-256, 호스트명 제외)
.github/workflows/            ci.yml(Rust·Python, macOS·Linux, Python 3.10·3.14), release.yml(태그 기반, 승인 환경 필요)
LICENSE-MIT, LICENSE-APACHE   MIT OR Apache-2.0 (0015 기본값 적용), README.en.md(PyPI용 영어 설명)
```

### 실행한 명령과 결과

| 검증 | 명령 | 결과 |
|---|---|---|
| Rust 포맷 | `cargo fmt --all --check` | ✅ |
| Rust 린트 | `cargo clippy --workspace --all-targets -- -D warnings` | ✅ 경고 0 |
| Rust 테스트 | `cargo test -p memopro` | ✅ 1 통과 |
| 확장 빌드·설치 | `VIRTUAL_ENV=.venv maturin develop --release` | ✅ abi3 wheel (Python ≥ 3.10) |
| Python 테스트 | `pytest -q` | ✅ 4 통과 |
| Python 린트·포맷 | `ruff check` · `ruff format --check` | ✅ (자동 수정 5건 후) |
| wheel 빌드 | `maturin build --release --out dist` | ✅ `memopro-0.0.1-cp310-abi3-macosx_11_0_arm64.whl` (195,640 B) |
| sdist 빌드 | `maturin sdist --out dist` | ✅ `memopro-0.0.1.tar.gz` (11,095 B) |
| PyPI 메타데이터 | `twine check dist/*` | ✅ 둘 다 PASSED |
| 새 가상환경에 wheel 설치 | `pip install --no-index --find-links dist memopro` → import | ✅ `version 0.0.1 core 0.0.1` |
| 새 가상환경에서 sdist 컴파일 설치 | `pip install dist/memopro-0.0.1.tar.gz` → import | ✅ (wheel이 없는 플랫폼의 설치 경로 확인) |
| crates.io 패키징 | `cargo publish -p memopro --dry-run` | ✅ 패키징·검증 통과, **업로드는 dry-run으로 중단** |

### 개발 환경

| 항목 | 값 |
|---|---|
| 하드웨어 | MacBook Air (MacBookAir10,1), Apple M1, 8GB, 페이지 크기 16KB |
| OS | macOS 26.6.2 (Darwin 25.6.0) |
| Rust | rustc 1.98.1, cargo 1.98.1, pyo3 0.29.2 |
| Python | CPython 3.14.0 (GIL 빌드), maturin 1.15.0, pytest 9.1.1, ruff 0.16.8, twine 7.0.0, pip 26.2.1 |

## 결과 / 결론

- 걷는 뼈대가 완성되었다. 두 레지스트리로 가는 경로(PyPI wheel·sdist, crates.io 크레이트)가 **로컬에서 모두 관통**한다.
- **배포는 하지 않았다.** 공개 배포(0.0.1 이름 선점 포함)는 사용자 확인이 필요하다(0015).

## 한계 및 향후 과제

- CI는 워크플로 파일만 작성했다. 원격 저장소가 없어 GitHub Actions로는 실행하지 못했고, 같은 명령을 로컬(macOS arm64, Python 3.14)에서만 확인했다. **Linux와 Python 3.10은 미검증**이다.
- `pyo3/extension-module` 기능은 maturin 설정에서 켠다. PyO3 최신 버전의 권장 방식이 바뀌면 조정한다.
- 크레이트의 설명은 "초기 개발"로 명시했다(0013 V13).

## 논문 매핑

- **Artifact**: 재현 가능한 빌드 절차와 환경 기록 하네스는 아티팩트 평가 자료가 된다.
