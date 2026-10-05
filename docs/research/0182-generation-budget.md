# 0182. 생성도 예산이 덮는다: `memopro.generate`가 초안·캐시·중간값 몫을 덜어 둠

- **날짜**: 2026-10-05
- **유형**: implementation
- **상태**: 확정. E041(0183)로 검증
- **관련 기록**: 0165 (학습의 `hold_back`), 0175 (CPU), 0174 (E038b: 7B 생성 footprint +3,015MiB가 1.5GiB 예산 위), 0179 (채점표 A13)

## 사용자 지시

> "설치 허락할게, CI 통과하면 병합하고 1~3번 순서대로 진행해." 3번 = 0179의 "생성 예산(`hold_back`), 2,048토큰".

## 문제

- 학습은 예산이 단계 전체를 덮는다(0165). 생성은 그렇지 않았다.
- E038b에서 7B 생성의 프로세스 증가는 +3,015MiB였다. 예산 1.5GiB 위에 초안(1,226MiB), KV 캐시, 중간값이 얹혔다.
- 사용자가 `budget="2GB"`라고 하면 그 안에서 끝나야 목표(0148: 보장된 메모리 상한)와 맞다.

## 제작 (`python/memopro/llm.py`)

- `_generation_estimate(config, tokens)` = KV 캐시(2 × 층 × KV 머리 × 머리 차원 × 토큰 × 2B) + 한 층의 프롬프트 중간값(2B × 토큰 × (4h + 3i)) + 마지막 행의 float32 로짓.
- `memopro.generate`
  - 생성 전에 덜어 둔다: 1.5 × 추정(토큰 = 프롬프트 + `max_new_tokens`) + 초안 바이트(`memopro_draft_bytes`).
  - Apple GPU에서는 대상 모델의 forward가 끝날 때마다 `torch.mps.driver_allocated_memory() − 감싼 가중치`를 재서, 더 크면 몫을 올린다. 이 값에는 초안·캐시·중간값이 모두 들어 있다.
  - 가장 큰 가중치가 들어가지 않는 예산은 `BudgetExceeded`로 거절한다("raise the budget or lower max_new_tokens").
  - CPU는 추정만 쓴다(학습과 같다).
- `_hold`의 거절 문구는 무엇을 위한 몫인지(학습의 활성값 / 생성의 초안·캐시·중간값)를 말한다.
- **`memopro.rt.torch.generate`(낮은 수준)는 그대로**다. 덜어 두기는 한 줄 API의 약속이다.

## 결과에 주는 영향

- 같은 예산이면 가중치 몫이 초안 크기만큼 줄어 더 느려질 수 있다. 같은 프로세스 메모리로 비교하면 이전과 같다.
- 7B + 1.5B int4 초안은 예산이 약 2.3GiB 이상이어야 한다(가장 큰 가중치 1,039 + 초안 1,226 + 추정). E038b의 1.5GiB 설정은 이제 거절된다.

## 시험

- `test_generate_budget_covers_the_generation[cpu|mps]`
  - 한도가 줄어든다. 출력은 일반 탐욕 생성과 같다. 런타임 최대 계상 ≤ 원래 한도.
  - 6MiB 예산 + `max_new_tokens=4000`은 거절된다.

## 논문 매핑

- **System**: 메모리 상한 보장을 학습에서 생성까지(초안 모델 포함) 넓힘.
