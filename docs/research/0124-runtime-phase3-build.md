# 0124. 런타임 C-R 3단계 제작: C ABI(`mp_*`), Linux 투명 페이저(userfaultfd + 쓰기 보호), NumPy 연결, `load(fallback="stream")`, 경쟁 조건 수정

- **날짜**: 2026-10-03
- **유형**: implementation
- **상태**: 확정. CI(Linux) 검증과 관문 E027(G-R3, 0122) 실행 전
- **관련 기록**: 0122 (3단계 범위, E027 사전 등록), 0123 (G4: 3단계 마무리 뒤 전환), 0115 (2단계), 0109 (설계)

## 무엇을 만들었나

### C ABI — `crates/memopro-c` (`libmemopro_c`, `include/memopro.h`)

- 불투명 핸들 `mp_runtime`, `mp_pin`과 `mp_*` 함수.
  - 런타임: 만들기, 해제, 한도.
  - 버퍼: `mp_alloc`, `mp_add_file`, `mp_derive`(C 함수를 계산식으로).
  - 고정: `mp_pin_acquire`, `mp_pin_data`, `mp_pin_size`, `mp_unpin`.
  - 기타: `mp_prefetch`, `mp_evict`, `mp_free`, `mp_state`, `mp_nbytes`, `mp_stats_get`, `mp_predict`.
- **오류**: 정수 상태 코드(`MP_ERR_BUDGET` 등)와 스레드별 메시지(`mp_last_error`).
  - Rust 패닉은 경계를 넘지 않는다. `catch_unwind`로 막아 `MP_ERR_PANIC`으로 바꾼다.
- **고정은 배타적**이다. 쓰기 고정은 그 버퍼의 유일한 고정이라, C 코드가 쓰는 쪽과 읽는 쪽을 경쟁시킬 수 없다.
- **ABI 진화**: 통계·예측 구조체는 호출자가 채운 `size`로 버전을 맞춘다. `MP_ABI_VERSION` = 1.
- **시험**
  - Rust에서 C처럼 부르는 시험 4개: 예산보다 큰 파일의 비트 동일 왕복과 예측, 오류 코드와 메시지, C 계산 함수로 재계산, 버전.
  - C 연기 시험 `tests/smoke.c`: macOS에서 `ok`(최대 계상 7.0MiB / 한도 8.0MiB, 재읽기 24회, 재계산 1회, 쓰기 0). CI Rust 작업에도 넣었다.

### Linux 투명 페이저 — `memopro::rt::Pager` (`rt/pager.rs`, `rt/pager/imp.rs`)

- **`map(len)`**: userfaultfd에 등록한 익명 메모리를 돌려준다(MISSING + WP 모드). 코드는 평범한 load·store로 쓴다.
- **폴트 처리**: 페이저 스레드가 청크 단위로 처리한다(기본 1MiB). 처음 닿은 청크는 0, 이후에는 압축본을 풀어 `UFFDIO_COPY`로 채운다.
- **내보내기**: 청크를 쓰기 보호(쓰는 스레드는 기다림) → 무손실 압축(셔플 + zstd) → `MADV_DONTNEED`.
  - 압축 이득이 15% 미만이면 그 청크는 계속 둔다. 그래서 예산을 넘으면 `overruns`로 센다(폴트에는 거절할 길이 없음, 0122).
- **선택**: 폴트를 시계로 쓰는 재사용 거리. 메모리에 남은 청크는 폴트가 없어 "기한이 지난" 것으로 보여 남고, 방금 들어온 청크가 나간다(MRU).
- **권한**: `UFFD_USER_MODE_ONLY`(Linux 5.11+)로 권한 없이 연다. 안 되면 이유와 대안을 담은 `Error::Unsupported`.
  - 압축본이 풀리지 않으면(메모리 손상) 다른 데이터를 주지 않고 중단한다(abort).
- **macOS·Windows**: `Pager::new`가 `Unsupported`를 낸다(R7 플랫폼 정직성).
- **이 Mac에서 한 검증**: Linux 대상 타입 검사와 clippy(`-D warnings`)만. 코덱을 바꿔 끼운 작은 임시 크레이트에서 했다. **실제 동작은 CI의 Linux 시험 4개로 처음 확인한다.**

### NumPy 연결과 CLI

- **바인딩** `_core.RtPager`, `_core.numpy_set_handler`
  - NumPy의 메모리 처리기 API(`PyDataMem_SetHandler`, C API 표 304번, 기본 처리기 306번)로 연결한다.
  - 임계값 이상의 배열(기본 16MiB)은 페이저에서, 나머지는 NumPy 기본 할당기에서 받는다.
  - `realloc`·`free`도 처리한다. 처리기 캡슐이 페이저를 살려 두므로, 배열이 남아 있는 동안 페이저가 사라지지 않는다.
- **`memopro.rt.transparent(budget, threshold=, chunk=)`**: 블록 안에서 만든 큰 NumPy 배열을 투명하게 페이징한다. Linux가 아니면 `ModeUnavailable`.
- **`memopro run --transparent BUDGET script.py`**: 스크립트를 고치지 않고 적용한다. `--report-json PATH`는 보고와 페이저 계수를 JSON으로 남긴다(E027이 씀).

### 접근 계층 연결 — `memopro.load(..., fallback="stream")`

- 무손실로 들어가는 구성이 없을 때 명시적으로 고르면, `memopro.rt.torch.stream_model`로 저장 dtype 그대로 CPU에서 흘려 쓴다. 경고와 보고(`load.stream`)를 남긴다.
- `BudgetExceeded`의 제안에 `fallback='stream'` 줄을 더했다.
- 설정 `fallback`에 `"stream"`을 추가했다(`load` 전용; `optimize`에서는 `none`처럼 동작).

### 고친 것

1. **경쟁 조건(2단계부터 잠재)**
   - 고정이 버퍼를 되돌리려고 자리를 만드는 동안 압축 때문에 잠금이 풀린다. 그 사이 미리 읽기 스레드가 같은 버퍼를 먼저 되돌리면, 압축 상태를 가정한 코드가 패닉했다(`unreachable`).
   - 파이썬 시험 `test_memory_buffers_are_compressed_losslessly`가 이번에 드러냈다.
   - 고친 것: `restore_source`·`restore_packed`가 자리를 만든 뒤 상태를 다시 보고, 바뀌었으면 "되돌리지 않았음"을 돌려준다. 호출자가 다시 본다.
   - 같은 기회에 `make_room`이 자기가 자리를 만드는 대상 버퍼의 "작업 중" 표시를 기다리지 않게 했다. 재계산 경로에서 자기 자신을 기다리며 멈출 수 있었다.
   - Rust 회귀 시험 `prefetching_and_pins_race_on_compressed_buffers_safely`를 더했다.
2. **`stream_model(model_class=...)`**: 구체 모델 클래스에는 `from_config`가 없어 `_from_config`를 쓴다(`load(fallback="stream")`가 드러냄).
3. **기록 번호**: 3단계 코드 주석의 기록 번호를 0124로 맞췄다. 0123은 G4 결정에 썼다.

## 시험 (macOS M1)

- Rust: `memopro` 59개(1개 무시), `memopro-c` 4개, clippy·fmt 통과.
- C 연기 시험 `ok`.
- Python 302개 통과. 2개 건너뜀: Linux 전용 투명 모드 시험. 그 밖에 다른 OS 거절 시험과 `fallback="stream"` 시험 추가. ruff 통과.
- CI에 더한 것
  - Rust 작업에 `cargo test -p memopro-c`와 C 연기 시험.
  - 수동 작업 흐름 `.github/workflows/e027.yml`(E027).

## 한계

- 투명 모드는 압축만 한다. 압축되지 않는 데이터는 예산을 넘는다.
- NumPy 배열만 대상이다. C/C++ 프로그램은 `mp_*`로 명시적으로 쓴다.
- user-mode-only에서는 커널이 그 메모리에 쓰는 경우(`read(2)`로 직접 채우기 등) `EFAULT`가 난다.
- fork한 자식 프로세스에서는 페이저가 동작하지 않는다(문서화만).

## 논문 매핑

- **Implementation**: 쓰기 보호 기반의 안전한 내보내기, 폴트를 시계로 쓰는 재사용 거리, 처리기 API로 코드 변경 없이 NumPy 연결, C ABI의 오류·수명 모델.
- **Threats**: 멀티스레드 런타임의 경쟁 조건을 시험이 늦게 드러낸 사례와 그 수정.
