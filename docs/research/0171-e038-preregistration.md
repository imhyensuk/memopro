# 0171. E038 사전 등록: 8GB M1에서 Qwen2.5-7B 무손실 생성, 1.5B int4 초안으로 2.5배 이상

- **날짜**: 2026-10-05
- **유형**: experiment (사전 등록)
- **상태**: 확정. 실행 전
- **관련 기록**: 0170 (어휘 패딩), 0145·0152 (E033c 설계·결과), 0161 (7B 학습)

## 설계

- E033c와 같다. 실행기는 E033c를 복사해 대상·예산·경우·길이만 바꿨다(`experiments/e038_7b_generation/run.py`).
  - 환경 변수(`MallocLargeCache=0`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1`, 오프라인), 경우마다 새 프로세스, 스왑 오염 규칙, 판정 코드, 기준.
- **대상**: Qwen2.5-7B-Instruct bf16(15,231,233,024바이트), 리비전 `a09a354…`, `stream_model(budget=2GiB, device="mps")`.
- **경우**: P(일반 `generate`, 기준), S15(`rtt.generate` + Qwen2.5-1.5B int4 초안). 3B 자기 초안에 해당하는 7B int4 자기 초안은 메모리 때문에 뺐다.
- **프롬프트**: E033c의 앞 2개(설명, 코드). 각 32토큰, 탐욕. 일반 7B 생성이 토큰당 약 9초라서 줄였다.
- **명령**: `caffeinate -i .venv/bin/python -m experiments.e038_7b_generation.run --all`. 결과는 `docs/research/data/e038/`.

## 기준 (E033c와 같다)

| # | 기준 |
|---|---|
| **X1** | 두 프롬프트 모두 S15의 32토큰이 P와 같다 |
| **X2** | S15 토큰당 시간 ≤ P의 1/2.5 |
| **X3** | S15 footprint 증가 ≤ 예산 + 초안 바이트 + 512MiB, 그리고 ≤ 7B 크기의 절반(7,262MiB). 스왑 증가 ≤ 64MiB(오염 규칙 뒤), 런타임 최대 계상 ≤ 한도 |

- 실패하면 멈추고 보고한다.

## 논문 매핑

- **Evaluation (G4 추론)**: 8GB 기기에서 7B 무손실 생성의 토큰당 시간. 초안이 없을 때와 있을 때.
