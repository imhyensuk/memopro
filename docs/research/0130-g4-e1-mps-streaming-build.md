# 0130. G4 E1 제작: 런타임 메모리를 Mac GPU가 복사 없이 쓴다 — 1.5B LoRA 한 단계 257초 → 약 5초

- **날짜**: 2026-10-03
- **유형**: implementation + 개발 점검(사전 등록 측정 아님)
- **상태**: 확정. 사전 등록 E028(0131) 전
- **관련 기록**: 0125 (G4 설계, E1), 0126 (방향 승인), 0115 (CPU 스트리밍), 0121 (E026: CPU LoRA 257초/단계), 0067·0072 (E015 G-F: 무복사 Metal 버퍼), 0080 (MPS 힙)

## 배경

- G4의 승부처인 공학 트랙 E1이다. 런타임이 붙잡은 메모리를 Apple GPU(MPS)가 그대로 읽게 해서, CPU 경로의 느린 계산(0115: M1 bf16 역전파 2.8 GFLOPS)을 피한다.
- 먼저 M1 GPU의 bf16 성능을 쟀다(1,536 × 8,960, 128행).

| 연산 | CPU(0115) | MPS |
|---|---|---|
| 순전파 행렬곱 | 120 GFLOPS | **643 GFLOPS** |
| 역전파 `grad @ W` | 2.8 GFLOPS | **259 GFLOPS** |
| 1행 생성(행렬-벡터) | — | 0.5ms(약 55GB/s) |

## 무엇을 만들었나

- **Rust**: `residency::metal_wrap(ptr, len)` / `metal_release(buffer)`.
  - 쪽 정렬된 아무 메모리나 `newBufferWithBytesNoCopy`로 감싼다. 0072의 파일 매핑 전용 코드를 일반화했다.
  - 바인딩: `_core.metal_wrap`, `_core.metal_release`, `RtPin.address`.
- **`memopro.rt.torch.stream_model(..., device="mps")`**
  - `_MetalPins`: 고정한 가중치마다 무복사 `MTLBuffer` → DLPack(kDLMetal) → MPS 텐서. 런타임 영역은 쪽 정렬 mmap이라 그대로 감쌀 수 있다.
  - **수명 규칙(GPU가 아직 읽는 메모리를 런타임이 옮기지 않도록, R4)**
    1. torch가 저장소를 놓으면(모든 뷰가 사라짐) DLPack deleter가 그것을 적어 두기만 한다.
    2. 다음 안전 지점(다음 고정, 역전파의 참조 복원, `finish()`)에서 이미 큐에 들어간 작업 뒤에 MPS 이벤트를 기록한다.
    3. 그 이벤트가 끝난 뒤에야 버퍼와 고정을 돌려준다.
    4. 예산이 차서 고정이 거절되면 GPU를 기다려 돌려주고 한 번 더 시도한다.
  - 모델 버퍼(회전 위치 등)는 GPU로 옮기고, `model.device`는 mps를 돌려준다.
  - 학습: `saved_weights`의 참조 저장·복원이 MPS에서도 같다(복원 때 다시 고정해 감쌈).
  - `StreamedWeights.finish()`: GPU를 기다리고 모든 고정을 돌려준다.

## 개발 점검

- **정확성(작은 GPT-2, 9MiB 예산)**: MPS 스트리밍이 **같은 GPU에서 보통으로 불러온 모델과 비트 동일**했다.
  - 로짓, 탐욕 생성 8토큰, LoRA 3단계의 손실과 학습된 어댑터가 모두 같았다.
  - 최대 계상 4.98 / 한도 4.98MiB, `finish()` 뒤 고정 0.
  - 시험 `test_mps_streaming_equals_the_model_loaded_on_mps`(MPS가 없으면 건너뜀)를 추가했다.
- **Qwen2.5-1.5B, 768MiB, MPS**(`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1`, `MallocLargeCache=0`)

| 작업 | MPS 스트리밍 | CPU 스트리밍(E026, 같은 예산) |
|---|---|---|
| LoRA(q·v, r=8), 1 × 128토큰, 한 단계 | **6.2 / 5.3 / 4.8초** | 257초 |
| 탐욕 생성 | 1.86초/토큰(첫 토큰 3.95초) | 1.34초/토큰 |

- **해석**
  - **학습이 약 50배 빨라졌다.** 계산이 GPU로 가서, 이제 단계 시간은 디스크 다시 읽기(정방향·역방향 각 약 2.3GB)가 정한다.
  - **생성은 오히려 느리다.** 생성은 읽기가 주도하는데, 토큰마다 약 340개 가중치를 감싸고 이벤트로 기다리는 비용이 더해진다. 생성 속도는 E4(int4 초안 추측 디코딩)와 버퍼 재사용으로 다룬다.
  - **메모리**: 이 실행에서 시스템 스왑이 약 1.1GB 늘었다. 프로세스 RSS는 968MB였다. 하지만 GPU 할당(Metal 힙)은 RSS에 잡히지 않아 원인을 가릴 수 없다.
    - 그래서 사전 등록 실험은 `proc_pid_rusage`의 `phys_footprint`(GPU 몫 포함)를 쓴다.
    - 확인: MPS에 512MiB를 잡으면 RSS는 324MiB로 보이지만 footprint는 675MiB로 보인다.

## 시험

- Python 303개 통과(Linux 전용 2개 건너뜀), Rust 59개 통과, clippy·fmt·ruff 통과.

## 한계

- 생성 경로의 감싸기·동기화 비용.
- MPS의 bf16 연산은 CPU와 비트가 다르다. 무손실 비교는 같은 장치끼리 한다.
- 활성값 메모리는 아직 런타임이 관리하지 않는다(E3).

## 논문 매핑

- **Implementation**: 통합 메모리에서 런타임 소유 메모리를 GPU가 무복사로 쓰는 방법과, DLPack deleter + GPU 이벤트로 지키는 수명 규칙.
- **Evaluation 예고**: 8GB M1에서 16비트 LoRA 단계 시간 50배 단축(E028에서 사전 등록 측정).
