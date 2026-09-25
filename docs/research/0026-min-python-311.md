# 0026. 최소 Python 3.10 → 3.11 상향과 3.10 흔적 정리

- **날짜**: 2026-09-25
- **유형**: decision
- **상태**: 확정
- **관련 기록**: 0025 (RS2 복사 없는 입력, 설계 영향 1), 0015 (3.10 기본값 적용), 0016 (S1 당시 cp310 빌드)

## 사용자 결정

> "파이썬 버전을 업데이트하고 이전 버전들을 정리해줘."

해석: 최소 지원 버전을 0025에서 제안한 **3.11**로 올리고, 프로젝트 설정·문서·빌드 산출물에 남은 3.10 흔적을 정리한다. 기기에 설치된 Python은 건드리지 않는다.

## 이유

- abi3(여러 Python 버전에서 쓰는 단일 wheel) 조건에서 **buffer protocol은 3.11부터** 쓸 수 있다. 텐서 메모리를 복사 없이 Rust로 넘기는 경로(0025 RS2)의 전제 조건이다.
- Python 3.10은 2026년 10월 지원 종료 예정이다.
- 3.11부터 표준 라이브러리 `tomllib`을 쓸 수 있어, 테스트의 정규식 우회 코드를 없앴다.

## 변경 내용

| 대상 | 이전 | 이후 |
|---|---|---|
| `pyproject.toml` `requires-python` | `>=3.10` | `>=3.11` |
| `pyproject.toml` ruff `target-version` | `py310` | `py311` |
| `crates/memopro-py/Cargo.toml` PyO3 | `abi3-py310` | `abi3-py311` |
| `.github/workflows/ci.yml` 매트릭스 | 3.10, 3.14 | 3.11, 3.14 |
| `tests/test_skeleton.py` | 정규식으로 Cargo.toml 파싱 | `tomllib` |
| README, architecture.md §10 | 3.10 | 3.11 |
| `dist/` 빌드 산출물 | `cp310-abi3` wheel | 삭제 후 `cp311-abi3` wheel·sdist로 재빌드 (git 추적 대상 아님) |
| 과거 연구 기록 0015·0016 | — | 내용은 당시 기록으로 보존하고 상태 줄에만 변경 표시 |

- ruff가 3.11 기준으로 `UP017`(`datetime.UTC` 사용 권고)을 새로 표시했다. 대상인 `experiments/_harness/env.py`의 해시는 이미 완료된 실험(E006~E008)의 env.json에 기록되어 있다. 그래서 파일을 고치지 않고 `experiments/**`에 대한 예외로 처리했다. 과거 기록의 해시 일치를 확인했다.

## 검증

| 검사 | 결과 |
|---|---|
| `cargo clippy -D warnings`, `cargo test -p memopro` | ✅ 7 통과 |
| `ruff check`, `ruff format --check` | ✅ |
| `maturin develop` + `pytest` | ✅ 4 통과 (tomllib 사용) |
| wheel·sdist 빌드, `twine check` | ✅ `memopro-0.0.1-cp311-abi3-macosx_11_0_arm64.whl` |
| 새 가상환경 설치 | ✅ 메타데이터 `Requires-Python >=3.11` |
| 과거 실험 env.json의 모듈 해시 | ✅ 모두 일치 |

## 한계

- 이 기기에는 Python 3.14만 있다. **3.11 환경에서의 실제 동작은 CI(원격 저장소 연결 후)에서 확인**해야 한다.
- 복사 없는 입력(RS2)의 구현은 RS1~RS8 승인 뒤에 진행한다. 이번 변경은 그 전제 조건만 갖춘 것이다.
