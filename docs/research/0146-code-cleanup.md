# 0146. 코드 정리: 쓰이지 않는 코드 제거, 중복 통합, 작업 트리를 더럽히던 시험 수정

- **날짜**: 2026-10-04
- **유형**: implementation (정리)
- **상태**: 확정
- **관련 기록**: 0091·0092 (Colab 노트북 생성기), 0130·0136 (MPS 무복사 텐서), 0072 (file-backed 상주)

## 사용자 지시

"전체적인 코드의 깔끔함을 정리해봐. 불필요한 파일, 코드 등은 절약하던가 대체, 제거해."

## 범위와 원칙

- **손대지 않은 것**
  - `docs/research/`의 기록과 데이터: 과거 기록은 고치지 않는다.
  - `experiments/`의 실험 스크립트: 해시가 `env.json`에 남아 있어 고정이다.
  - 공개 API. Rust의 `Pinned::is_writable`은 안에서 쓰이지 않지만, 공개 타입의 접근자라 남겼다.
- **찾는 방법(패키지 설치 없이)**
  - 파이썬: 패키지의 모든 최상위 함수·클래스·상수 이름이 `python/`, `tests/`, `experiments/`, `examples/` 어디서 쓰이는지 셌다.
  - Rust: 모든 `pub` 항목이 정의 밖에서 쓰이는지 셌다.

## 바꾼 것

| 구분 | 내용 |
|---|---|
| 제거 | `hibernate/_methods.py::drop_kept`: 어디서도 부르지 않았다. 남은 spill 파일은 깨울 때와 종료 때(`_ssd._cleanup`) 지워진다 |
| 제거 | `hibernate/_tensors.py::MIN_TENSOR_BYTES`: 쓰이지 않는 상수 |
| 통합 | MPS 무복사 텐서의 DLPack 조립이 `residency.Mapping`과 `rt/torch.py::_MetalPins`에 따로 있었다. `residency.metal_tensor(buffer, nbytes)` 하나로 합쳤다(같은 MTLBuffer를 두 번 가져오지 말라는 0136의 경고를 문서에 담음) |
| 통합 | safetensors 파일 찾기(로컬 폴더 또는 HF 캐시)가 `residency._safetensors_files`와 `rt/torch.py::_files`에 거의 같게 있었다. `access/_info.local_safetensors`로 합쳤다. 내려받기 대체 경로(`residency`만 해당)는 그대로 둔다 |
| 통합(시험) | 작은 GPT-2를 만드는 함수 5개(`test_access`, `test_e021_fixes`, `test_e022_fixes`, `test_defaults_da_dd`, `test_hibernate`)를 `tests/helpers.py::gpt2(dtype, dropout=, **overrides)`로 합쳤다. 각 파일의 설정(드롭아웃 등)은 그대로다 |
| 수정(시험) | `test_colab_notebook.py`가 시험 때마다 `examples/colab_t4_*.ipynb` 6개를 다시 써서(커밋 해시 줄) 작업 트리를 더럽혔다. `build.py`에 `--nb-dir`를 더해 시험은 임시 폴더에 만들어 비교한다 |
| 삭제(추적 안 함, 다시 만들 수 있음) | Rust 빌드 산출물 `target/`(612MB, `cargo clean`), `__pycache__`·`.DS_Store` 24개, `.pytest_cache`, `.ruff_cache`, `experiments/e015_foundations/native/build`(52KB) |

- **남긴 것(추적 안 함)**: `experiments/*/_scratch/*.log`(32KB). 옛 실험의 실행 로그라 다시 만들 수 없어 지우지 않았다.

## 확인

- `pytest` 310 통과, 2 건너뜀. `ruff check`·`ruff format --check`(python, tests, experiments)도 깨끗하다.
- 시험 뒤에도 추적 파일이 바뀌지 않는다(`git status`에서 노트북 변경 없음).
- Rust 코드는 바꾸지 않았다. 그래서 `target/`을 다시 채우는 Rust 검사는 CI에 맡겼다.

## 제안(이번에 하지 않음)

- `CLAUDE.md`의 연구 이력 문단이 매우 길다(0001~0145 요약). 세션마다 읽히므로 줄일 수 있다. 하지만 프로젝트 지침이라 사용자 결정이 필요하다.
- 시험 파일 이름 일부가 기록 번호를 따른다(`test_e021_fixes.py`, `test_review_0100.py`, `test_defaults_da_dd.py`). 기능 이름으로 바꿀 수 있다. 하지만 기록에서 그 이름으로 인용하므로 그대로 두었다.

## 논문 매핑

- 해당 없음(공학 정리). 재현성: 시험이 작업 트리를 바꾸지 않게 되어 실험 커밋의 상태가 깨끗해졌다.
