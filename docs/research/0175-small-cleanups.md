# 0175. 작은 정리 둘: CPU에서도 예산이 학습 단계를 덮음, `memopro run` 아래의 흘려 쓰는 MPS 모델은 워터마크 0.01

- **날짜**: 2026-10-05
- **유형**: implementation (+ 개발 측정)
- **상태**: 확정
- **관련 기록**: 0165 (`hold_back`, MPS만), 0169 (E037c), 0162 (워터마크 0.01, `memopro run`은 미결정), 0080·0081 (`memopro run`의 0.1)

## 사용자 지시

"우선 2번과 6번을 진행해." 6번은 앞 답변의 "작은 정리"다.
- `memopro run`의 워터마크(0.1 → 0.01?)
- CPU에서도 예산이 학습 단계 전체를 덮게

## 1. `memopro run`과 흘려 쓰는 MPS 모델의 워터마크

- **문제**
  - `memopro run`은 스크립트를 다시 시작하면서 `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1`을 넣는다(0081).
  - `stream_model(device="mps")`는 값이 없을 때만 0.01을 넣는다(0162). 그래서 `memopro run` 아래에서는 0.1이 남는다.
  - 0162의 개발 측정에서 3B·512토큰은 0.1일 때 예산 + 1,332MiB, 0.01일 때 + 515MiB였다.
- **`memopro run`을 0.01로 바꾸지 않은 이유**
  - 0.1은 불러온 모델(int4·bf16)에서 잰 값이다(E019, 0080). 그 실험에서 0.01은 1.5B int4가 느려지는 경향이 있었다(11.0·11.7 대 12.0~13.9토큰/초, 변동이 큼).
  - 두 측정이 서로 다른 경우를 지지하므로, 각 경우에 맞는 값을 쓴다.
- **제작**
  - `memopro run`이 다시 시작할 때 `MEMOPRO_RUN_MPS_WATERMARK=0.1`도 같이 넣는다(`memopro.env.MPS_LOW_WATERMARK_BY_RUN`).
  - `stream_model(device="mps")`은 값이 없거나 그 값이 `memopro run`이 넣은 값과 같으면 0.01로 바꾼다. 사용자가 직접 정한 값은 그대로 둔다.
- **시험**: `test_streamed_mps_models_lower_the_ratio_run_set_but_not_a_user_value`
  - 실제 `memopro run`으로 스크립트를 돌려, `stream_model` 뒤의 값이 0.01인지, 사용자 값 0.7은 남는지 본다.
  - 수정을 되돌리면 실패하는 것을 확인했다.

## 2. CPU에서도 예산이 학습 단계를 덮음

- **문제**: `finetune`의 활성값 몫(`hold_back`)은 MPS에서만 했다(0165). CPU에서는 활성값이 예산 위에 그대로 얹혔다.
- **제작** (`python/memopro/llm.py`)
  - CPU에서도 첫 단계 전에 활성값 추정의 1.5배를 덜어 둔다. 가장 큰 가중치가 들어가지 않는 예산은 MPS와 같이 거절한다.
  - CPU에는 MPS의 `driver_allocated_memory` 같은 할당기 계수가 없다. 그래서 단계마다 재서 늘리는 것은 MPS만 하고, CPU는 추정을 그대로 쓴다.
  - 시험 `test_one_line_finetune_and_generate`(CPU)의 예산을 9MiB → 24MiB로 올렸다. 9MiB는 가중치만 덮던 값이라 이제 거절된다. 24MiB에서도 한도가 모델(약 12MB)보다 작아 흘려 쓰기는 그대로 시험된다. `held_back > 0`도 확인한다.

### 개발 측정 (사전 등록 결과가 아님)

| 항목 | 값 |
|---|---|
| 기계 | M1 8GB, 전원 연결, 커밋 `23e5903` + 이 변경(작업 트리) |
| 과제 | Qwen2.5-1.5B-Instruct bf16, CPU, 예산 1GiB, `seq_len=256`, E037 글 첫 510토큰 → 2단계, 기본 LoRA·체크포인팅 |
| 측정 | 경우마다 새 프로세스, `MallocLargeCache=0`, 기준 footprint는 torch·transformers·peft를 불러온 뒤 |
| 비교 | `on`(이 변경) / `off`(`llm._hold`를 아무것도 안 하게 바꿔 이전 동작 재현) |
| 스크립트·원시 결과 | `docs/research/data/dev0175/dev.py`, `out.jsonl` |

| 경우 | footprint 증가 | 덜어 둔 몫 | 런타임 한도 / 최대 계상 | 단계 시간(초) | 시스템 스왑 증가 |
|---|---|---|---|---|---|
| on | **+942MiB(≤ 예산 1,024)** | 176MiB | 843 / 843MiB | 551, 527 | +294MiB |
| off | +1,113MiB(예산 + 89) | (계산만, 적용 안 함) | 1,019 / 1,019MiB | 534, 534 | −184MiB |

- 손실은 두 경우가 비트 단위로 같았다(`0x1.7f2a0a…p+1`, `0x1.911ed2…p+1`).
- 덜어 두면 프로세스가 예산 안에 든다. 속도 차이는 측정 변동 안이다.
- `on`의 시스템 스왑 +294MiB는 먼저 돈 경우였고 다음 경우에서 줄었다. 0152에서 본 것처럼 기계 상태를 반영한 것으로 보지만, 반복하지 않았으므로 단정하지 않는다.
- CPU 학습은 bf16 backward가 느린 경로를 타서(0115) 1.5B 256토큰 한 단계가 약 9분이다. 메모리 상한은 지켜지지만 실용 속도는 아니다.

## 확인

- Python 전체 시험, ruff, Rust 시험·fmt·clippy는 PR 전에 돌린다(아래 커밋).

## 논문 매핑

- **System**: 메모리 상한 보장을 장치(GPU/CPU)와 실행 방식(`memopro run`)에 상관없이 지키기 위한 세부 처리.
- **Threats**: CPU의 활성값 몫은 측정이 아니라 추정이다.
