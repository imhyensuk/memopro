# 0163. E037 사전 등록: `finetune` 기본 길이(512토큰)에서 메모리 상한 — 3B·7B, 라이브러리 기본 설정 그대로

- **날짜**: 2026-10-05
- **유형**: experiment (사전 등록)
- **상태**: 확정. 실행 전
- **관련 기록**: 0162 (워터마크 기본값), 0161 (S1, 129토큰), 0136

## 설계

- **경우**(각각 새 프로세스, 스왑 오염 규칙)

| 이름 | 모델 | 예산 |
|---|---|---|
| T3a | Qwen2.5-3B-Instruct | 1GiB |
| T3b | Qwen2.5-3B-Instruct | 768MiB |
| T7a | Qwen2.5-7B-Instruct(리비전 `a09a354…`) | 2GiB |
| T7b | Qwen2.5-7B-Instruct | 1.5GiB |

- **과제**: `memopro.finetune(model, texts, seq_len=512)`(기본 LoRA, 체크포인팅 켬).
  - 글은 WikiText-2 앞부분 511토큰 × 5개다(`docs/research/data/e037/texts.json`). 끝 토큰을 더해 한 단계가 512토큰이고, 5단계다.
- **환경**
  - 실행기는 `PYTORCH_MPS_LOW_WATERMARK_RATIO`를 **넣지 않는다**(라이브러리 기본값을 시험한다). `MallocLargeCache=0`, 오프라인.
  - 기준 footprint는 torch·transformers·peft를 불러온 뒤, **MPS를 시작하기 전에** 잰다. MPS 초기화(약 11MiB)는 증가에 들어간다.
- **명령**: `caffeinate -i .venv/bin/python -m experiments.e037_long_seq.run --all`. 결과는 `docs/research/data/e037/`.

## 기준

| # | 기준 |
|---|---|
| **K1** | 네 경우 모두 5단계 완주 |
| **K2** | 모델마다 두 예산의 손실이 모두 유한하고 비트 동일 |
| **K3** | 모든 경우 footprint 증가 ≤ 예산 + 512MiB, 스왑 증가 ≤ 64MiB(오염 규칙 뒤), 런타임 최대 계상 ≤ 한도 |

- 실패하면 멈추고 보고한다.
- **보고**: 단계 시간, 토큰/초(단계 2~5), 적용된 워터마크 값.

## 논문 매핑

- **Evaluation (G4)**: 실사용 길이(512토큰)에서의 메모리 상한과 무손실. 3B·7B, 8GB M1.
