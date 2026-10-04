# 0149. G4 E8 제작: 한 줄 API `memopro.finetune` / `memopro.generate`와 빠른 시작

- **날짜**: 2026-10-05
- **유형**: implementation (+ 개발 확인)
- **상태**: 확정
- **관련 기록**: 0148 (목표, 성공 기준 S4), 0125 (E8 설계), 0133 (E3), 0139·0142 (E4), 0138 (G4-B1)

## 사용자 지시

"순서대로 전부 진행해."(0148 뒤 제안 순서: E033c → 한 줄 API → 7B → 경쟁자 비교)

- E033c는 다른 앱(Chrome, KakaoTalk)이 열려 있어 사전 등록 조건(0145)이 맞지 않는다. 그래서 다음 단계부터 진행했다.

## 선행·재사용

- LoRA는 **PEFT를 감싼다**(0010: 다시 구현하지 않음). 저장 형식이 표준 PEFT 어댑터라서 HF·PEFT 생태계와 호환된다(0125 E8).
- 흘려 쓰기, 체크포인팅, 조각 손실, 초안, 행 불변 생성은 기존 부품(`memopro.rt.torch`)을 그대로 쓴다. 새 알고리즘은 없다.

## 제작 (`python/memopro/llm.py`, `memopro.finetune`·`memopro.generate`로 노출)

- `finetune(model, texts, *, tokenizer, budget, device="auto", epochs, seq_len=512, lr=2e-4, rank=8, alpha=16, targets=(q,k,v,o_proj), seed)` → `FinetuneResult(model, adapter, losses, seconds, tokens)`
  - `model`은 이름, 폴더, 또는 흘려 쓰는 모델이다. `device="auto"`면 MPS를 쓸 수 있을 때 MPS, 아니면 CPU다.
  - `peft.get_peft_model`을 쓴다.
    - **발견**: PEFT는 어댑터를 기본 가중치가 있는 장치에 둔다. 흘려 쓰는 모델에서는 그곳이 meta 장치다. 그래서 LoRA 층만 `to_empty(device)`로 실제 장치에 만들고, PEFT 자신의 초기화(`reset_lora_parameters`)를 다시 부른다.
    - 학습 매개변수는 float32다.
  - 글은 끝 토큰으로 이어 `seq_len` 조각으로 자르고, 조각 하나가 한 단계다.
  - 각 단계는 `saved_weights` + `causal_lm_loss`(조각 손실, 0136) + 재진입 체크포인팅 + AdamW + `finish()`다.
- `generate(model, prompt, *, tokenizer, budget, device, draft=None, max_new_tokens=256, chat=True)` → 문자열
  - 문자열 프롬프트에는 채팅 템플릿을 쓴다(있고 `chat=True`일 때).
  - `draft`는 이름, 폴더, 또는 `draft_model` 결과다. 생성은 `rtt.generate`로 하므로 초안이 있어도 없어도 일반 탐욕 생성과 같다.
- README·README.en 첫머리를 0148의 목표 문장과 이 API의 세 줄 예시로 바꿨다. README의 "1. 목표와 대상" 표도 0148로 바꿨다.
- **시험** `test_one_line_finetune_and_generate`(작은 Qwen2, CPU, 9MiB 예산)
  - 손실이 줄어든다. 기본 가중치는 학습되지 않는다. 어댑터가 저장된다(`adapter_config.json`).
  - `generate`가 같은 모델의 일반 `model.generate`와 같은 글을 낸다.
  - 예산과 고정을 지킨다.
- 하지 않은 것(YAGNI): 예상 메모리·시간 계획표, 배치 > 1, 검증 집합, 학습률 일정. 필요해지면 더한다.

## 개발 확인 (결과 아님, M1 8GB, MPS)

| 모델 | 예산 | 학습(seq 128, LoRA q·k·v·o) | 생성 24토큰: 초안 없음 → 1.5B int4 초안 | 출력 같음 | 최대 footprint 증가 |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 768MiB | 2에폭 8단계, 3.9초/단계(29.9토큰/초). 2에폭 손실이 1에폭보다 낮음 | 40.5 → 17.7초 | 예 | 2,455MiB(초안 포함) |
| Qwen2.5-3B | 1GiB | 4단계, 7.3초/단계(15.8토큰/초) | 82.9 → 44.6초 | 예 | 2,761MiB(초안 1,226MiB 포함) |

- 3B 학습은 E028c(LoRA q·v, 실험 코드, 약 10.5초/단계)보다 빠르다. 다만 텍스트와 조건이 달라 비교하지 않는다.
- 초안은 미세 조정되지 않은 기본 1.5B다. 그래도 출력은 미세 조정된 모델의 일반 생성과 같다(초안은 제안만 한다).

## 논문 매핑

- **System/Usability (G4 E8)**: 한 줄 API가 기존 부품을 묶는다. PEFT 표준 어댑터로 호환된다. Unsloth식 사용성과 비교하는 근거.
