# 0081. W1+W2 구현: `memopro run`이 MPS 저수위 비율 0.1로 다시 실행하고, doctor와 파일 기반 load가 안내

- **날짜**: 2026-09-28
- **유형**: implementation (사용자 결정)
- **상태**: 확정
- **관련 기록**: 0080 (E019: 추가 1GB는 PyTorch MPS 할당기의 1GiB 힙, 선택지 W1~W4), 0061 (F4: `MallocLargeCache=0` 재실행)

## 사용자 결정

> "PR #15 병합하고 W1+W2 적용해."

- **W1**: `memopro run`이 Apple silicon에서 `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1`로 자신을 다시 실행한다.
- **W2**: doctor와 `load(residency="file")` 보고에서 이 설정을 안내한다.
- W3(제어기 재측정)과 W4(학습 비용)는 진행하지 않았다.

## 구현

- **`memopro.env`**
  - 상수: `MPS_LOW_WATERMARK_VAR`, `MPS_LOW_WATERMARK = "0.1"`, `MPS_HEAP_NOTE`.
  - `mps_heap_reserve_on()`: macOS이고 `platform.machine() == "arm64"`이며 사용자가 변수를 정하지 않았으면 참이다.
  - 환경만 읽고 torch를 import하지 않는다(doctor(devices=False) 규칙).
- **`memopro doctor`**: `mps_heap_reserve_on()`이면 경고 목록에 `MPS_HEAP_NOTE`를 넣는다.
- **`load(residency="file")`**
  - 프로세스마다 한 번 `suggested` 항목을 남긴다(`_suggest_mps_heap_setting`).
  - hibernate의 malloc 안내와 같이 `applied` 항목보다 앞에 둔다. 그래서 마지막 항목은 계속 불러오기 결과다.
- **`memopro run`**
  - `_restart_without_malloc_cache`를 `_restart_for_macos`로 바꿨다.
  - `MallocLargeCache=0`(F4)과 저수위 비율을 **한 번의 재실행**으로 함께 설정한다.
  - 사용자가 이미 변수를 정했으면 그 값을 둔다.
  - `--keep-mps-heap`으로 끌 수 있다. `--dry-run`과 파이썬 안에서의 `main()` 호출은 재실행하지 않는다(기존과 같음).
- **정밀한 조건** (구현 중에 확인)
  - 비율 0.1은 **MPS에서 쓰는 메모리가 권장 최대치의 10%(8GB M1에서 546MiB)를 넘어야** 효과가 있다. 그 아래에서는 할당기가 압박 상태가 아니어서 여전히 1GiB 힙을 잡는다.
  - 빈 프로세스에서 20MiB 하나만 잡는 시험이 이 때문에 실패했다.
  - 실제 모델에서는 파일 매핑된 가중치가 드라이버 할당량에 들어가 이 조건을 넘는다(E019의 1.5B int4 1.2GB, 3B int4 2.1GB). 안내 문구에 이 조건을 넣었다.
  - 권장 최대치가 큰 기기(메모리가 많은 Mac)나 아주 작은 모델에서는 효과가 없을 수 있다.

## 시험

- **`tests/test_mps_heap.py`** (7개)
  - 탐지 조건(플랫폼, 아키텍처, 사용자 값).
  - doctor 안내.
  - 파일 기반 load 안내가 한 번만 나오는지, 사용자 값이 있으면 나오지 않는지.
  - 재실행: 비율만, 둘을 함께, 건너뛰기와 사용자 값 유지.
  - 실제 macOS에서 `memopro run`을 실행해 환경변수를 확인했다(기본 0.1, 사용자 값 0.7 유지).
  - **전제 확인 시험**: MPS가 할당 가능하면 권장 최대치의 12%를 먼저 잡은 뒤 20MiB를 할당한다. 기본 설정에서는 1GiB 이상, 0.1에서는 64MiB 미만이 늘어야 한다. torch 버전이 바뀌어 할당기 동작이 달라지면 이 시험이 알려 준다.
- **`tests/test_macos_malloc.py`**: 새 함수 이름과 인자(`keep_mps_heap`)에 맞췄다.
- **결과**
  - Python 238개 통과(이전 231개 + 새 시험 7개).
  - Rust 28개 통과.
  - ruff, cargo fmt, clippy(`-D warnings`) 통과.
  - clippy는 `VIRTUAL_ENV`를 지정해야 pyo3가 3.11 이상의 인터프리터를 찾는다.
- **끝에서 끝까지 확인** (1.5B int4, `residency="file"`, E017 채팅 프롬프트, 32토큰)
  - `memopro run`: `restarting with MallocLargeCache=0, PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1 ...`, footprint **461MiB**, 안내 없음.
  - `memopro run --keep-mps-heap`: footprint **1,470MiB**, 안내 한 번.

## 한계

- 학습 경로의 비용(W4)은 재지 않았다. `memopro run`은 학습 스크립트에도 이 설정을 켠다. E019에서 추론 속도 비용은 없었지만, 큰 임시 버퍼를 매 단계 잡는 학습에서는 할당 횟수가 늘 수 있다. 문제가 되면 `--keep-mps-heap`으로 끈다.
- 노트북이나 일반 스크립트에서는 memopro가 설정할 수 없다(MPS 시작 전이어야 함). 안내만 한다.
- `--keep-mps-heap`으로 끈 경우에도 안내 문구는 "`memopro run`이 대신 해 준다"는 문장을 그대로 보여 준다.

## 논문 매핑

- **논문 B (도구 논문)**: 코드 변경 없는 실행(`memopro run`)이 프로세스 수준 설정(할당기 캐시 2종)을 맞춘다. 8GB 기기에서 약 1GB 절감.
- **논문 C Threats to Validity**: 할당기 설정에 따라 footprint 측정이 1GB씩 달라진다.
