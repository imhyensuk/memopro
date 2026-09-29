# 0103. P-a, P-b 적용: 여유 계수 1.3, 학습 시작 때 CUDA 캐시를 비우고 실측, 가용량에는 통째로 빈 세그먼트만

- **날짜**: 2026-09-29
- **유형**: implementation (사용자 결정)
- **상태**: 확정. CUDA 재측정 전
- **관련 기록**: 0102 (E023 결과, F1·F2, 제안 P-a·P-b), 0099 (캐시 여유 포함, 여유 계수 1.25), 0100 (정밀 검토)

## 사용자 결정

> "PR #30 병합하고 P-a, P-b 적용해"

## 1. P-a: 여유 계수 1.25 → 1.3

- `access/_train.py::ALLOCATOR_MARGIN = 1.3`.
- 근거(E023 F1): 최대 예약에서 가중치·기울기·옵티마이저 상태를 뺀 양은 "micro × 측정한 1표본 활성값"의 1.27~1.29배였다(0.5B micro 2, GPT-2 micro 8, GPT-2 30% micro 2).
- E023의 네 계획에 1.3을 대입해도 선택은 8, 2, 2, 4로 그대로다(0102의 계산).

## 2. P-b: 캐시는 가정 대신 실측

### 설계

- **학습 세션 시작 (`train_session`)**
  - 모델이 CUDA에 있으면 예산을 재기 전에 `release_cuda_cache()`를 부른다. `torch.cuda.synchronize()` 뒤 `torch.cuda.empty_cache()`를 불러 캐시 할당기가 쓰지 않는 세그먼트를 드라이버에 돌려준다.
  - 텐서는 건드리지 않는다. 돌려준 양은 보고에 남긴다(`[applied] train_session - returned … of torch's unused CUDA cache before measuring`).
  - 실패하면 0을 돌려주고 그대로 잰다(fail-open). 결과는 작아질 뿐 틀리지 않는다.
- **가용량 측정 (`env/_torch.py::releasable_cache`)**
  - 0099는 캐시 여유 전체(reserved − allocated)를 가용으로 셌다. 이제는 **통째로 빈 세그먼트만** 센다.
  - 식: `reserved − active − inactive_split`. torch 캐시 할당기 통계(`memory_stats`)의 `reserved_bytes`, `active_bytes`, `inactive_split_bytes`를 쓴다.
  - 일부가 쓰이는 세그먼트의 빈 조각(`inactive_split`)은 그 조각에 맞는 할당에만 쓰인다. E023에서는 불러오기가 남긴 캐시 1.99GiB 가운데 1.67GiB가 끝까지 재사용되지 않았다.
  - 통계를 못 읽으면 0으로 본다(보수적).
  - 이 측정은 `load`, doctor, `check`, `optimize`에도 같이 적용된다. 이들은 캐시를 비우지 않는다(부작용 없음). 새 프로세스에서는 캐시가 거의 0이라 영향이 작다.
- **학습 세션의 흐름**: 캐시를 비운다 → 드라이버의 free가 그만큼 늘어난다 → 남은 캐시는 대부분 조각이라 세지 않는다 → 보유 메모리를 더해 예산을 잡는다(0099).

### 예상 영향 (E023 QLoRA 기준, 측정 아님)

- 세션 시작 때 캐시는 1.99GiB였다. 비울 수 있는 부분이 얼마인지는 이번 기록으로 알 수 없다.
  - 대부분 돌려받으면 예산은 E023과 비슷하고 micro 4를 유지한다.
  - 대부분 조각이면 예산이 약 1.8GiB 줄어 2+2가 된다. 그래도 OOM 없이 안전 쪽이다.
- 그래서 Colab 워커는 세션 진입 직후의 CUDA 상태(`session_measured`)를 추가로 기록한다. 시작 전 상태(`session_start`)와 비교하면 돌려받은 양과 남은 조각을 볼 수 있다.

## 시험

- 새 시험 3개(`tests/test_review_0100.py`)
  - 여유 계수가 E023 예약 배수(1.29) 이상이다.
  - `releasable_cache`는 통째로 빈 세그먼트만 세고, 통계가 없으면 0이다.
  - `release_cuda_cache`는 돌려준 양을 알려 주고, 실패하면 0이다.
- `tests/test_e022_fixes.py`의 보고 문구 검사를 `ALLOCATOR_MARGIN` 값에 맞췄다.
- 전체 시험 280개 통과. ruff check·format 통과. 노트북 5개를 다시 만들었다(워커 변경).
- CUDA 기기가 없어, 캐시를 비우는 경로는 로컬에서 실제로 돌리지 못했다. torch 함수를 흉내 낸 시험만 했다.

## 남은 것

- 7B QLoRA만 Colab에서 다시 재서 P-b의 실제 효과를 확인한다(0102의 권고).
  - 돌려받은 캐시, micro, 최대 예약이 예산 이하인지, 속도.

## 논문 매핑

- **논문 B Design**: 캐시 할당기 메모리를 "가용"으로 볼 때의 기준. 통째로 빈 세그먼트만 세고, 계획 전에 캐시를 비워 측정한다. 예약 기준 여유 계수의 실측 근거도 함께 싣는다.
