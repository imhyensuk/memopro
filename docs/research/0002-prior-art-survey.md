# 0002. 선행 기술 조사 및 중복성 검사

- **날짜**: 2026-09-24
- **유형**: survey
- **상태**: 확정 — 일부 정정됨(→ 0004: "예산 기반 자동 조합" 공백 판단이 과장됨)
- **관련 기록**: 0001, 0003

## 배경 / 동기

초기 기능 후보(메모리 추정, 블록 양자화, mmap 제로카피 로더, 레이어 스트리밍·오프로딩, 활성값 압축·체크포인팅,
메모리 풀 할당자)가 기존 생태계와 얼마나 겹치는지 확인하고, 이름 사용 가능 여부를 검증한다.

## 조사 방법

- 패키지 레지스트리 직접 조회: crates.io API, PyPI JSON API, PyPI Simple index(전체 약 89.8만 개 이름 정규식 검색), npm registry
- GitHub API (저장소·사용자 검색)
- 웹 검색 (2026-09 기준): 관련 라이브러리, 문서, 이슈, arXiv 논문

## 결과 1 — 이름 중복

| 대상 | 결과 (2026-09-24) |
|---|---|
| crates.io `memopro`, `memo-pro`, `memo_pro`, `memopro-py`, `memopro-rs`, `pymemopro` | 모두 미등록 (HTTP 404) |
| PyPI 동일 목록 | 모두 미등록 |
| npm `memopro` | 미등록 |
| PyPI 유사 이름 | `memoprop`, `memoproperty` (메모이제이션 유틸리티 추정, 혼동 위험 낮음) |
| GitHub 사용자 `memopro` | **존재** (2018-02 생성, 공개 저장소 0개) → `github.com/memopro` 사용 불가 |
| 동명 서비스 | "Memo Protocol"(memopro.ai, AI 에이전트 지식 프로토콜 — 조사 시점 도메인 미해석), MemoPro 암기 학습 앱 |

**시사점**: AI 분야에서 "memory"는 주로 에이전트·대화 기억(Memori, MemoRizz 등)을 가리킨다.
패키지 설명·키워드에 "GPU/RAM 하드웨어 메모리"를 명시해야 한다. 상표 조회(KIPO/USPTO)는 미실시.

## 결과 2 — 기능 중복

| 후보 기능 | 기존 도구 | 중복도 | 비고 |
|---|---|---|---|
| mmap 제로카피 로더 | safetensors [@safetensors] | 매우 높음 | Rust 코어 + PyO3 + maturin + memmap2 — memopro 계획과 기술 스택이 동일 |
| 가중치 양자화 (int8/int4/NF4) | bitsandbytes [@dettmers2023qlora], torchao [@torchao], HQQ, GPTQ, AWQ, GGUF | 포화 | Transformers가 20종 이상 양자화 방식을 기본 지원 |
| 8bit 옵티마이저 | bitsandbytes [@dettmers2022optimizers], torchao | 높음 | |
| 오프로딩·레이어 스트리밍 | accelerate, DeepSpeed ZeRO-Offload/Infinity [@rajbhandari2020zero; @rajbhandari2021infinity], FlexGen [@sheng2023flexgen], oLLM, air-rs(Rust+Python), StreamLoader(Rust, DLPack) | 높음 | Rust 구현 경쟁자도 존재 |
| 활성값 체크포인팅 | `torch.utils.checkpoint` [@chen2016sublinear] | 높음 | PyTorch 내장 |
| Rust 추론 엔진 | candle, mistral.rs, oxillama | 높음 | |
| 활성값 압축 | 연구는 다수, 유지되는 범용 라이브러리 적음 | 중간 | |
| **피크 메모리 예측** | `accelerate estimate-memory` [@accelerate_estimator]; 연구 프로토타입 [@gpumemest2026; @gpumempred2025dyn; @gpumemmm2025; @dnnabacus2022] | **낮음 (빈 자리)** | 아래 참고 |
| **예산 기반 자동 조합** | 해당 범용 도구 미발견 | **낮음 (빈 자리)** | |

### 피크 메모리 예측 분야 상세

- `accelerate estimate-memory`: meta device로 모델을 올려 가중치를 내려받지 않고 추정. 로딩량 중심이며 학습 시 추정은 단순 배수 수준.
  정확도 이슈가 보고됨 — 예: Llama-3.1-70B-Instruct에서 기대치 약 140GB 대비 약 64GB로 추정 (huggingface/accelerate#3379).
- 2025–2026 연구: 분석적 모델(레이어별 수식 합산, MAPE 약 8.7% 보고), 동적 분석, ML 기반 예측 등이 제안됨.
  한계로 새로운 아키텍처·GPU로의 일반화 부족, 메모리 최적화 기법 적용 후 절감량 반영 어려움, 침습적 통합 비용이 지적됨 [@gpumemest2026].
  조사 시점 기준 **이 분야를 대표하는 라이브러리는 확인되지 않음.**

## 결론

1. 이름 `memopro`는 crates.io·PyPI 모두 사용 가능.
2. 개별 메모리 절감 기법은 성숙한 도구가 이미 존재 → 재구현 시 차별성 낮음.
3. **"정확한 예측"과 "예산 기반 자동 설계"는 빈 자리** → 포지셔닝 결정은 0003.

## 한계 및 향후 과제

- 웹 검색은 미국 기준 결과이며, 비공개·신생 프로젝트는 누락 가능.
- 참고문헌 일부(arXiv 2025–2026)는 저자 정보 미확인 → references.bib에 `TODO` 표기, 논문 작성 전 원문 확인 필요.
- 상표 조회 미실시.
- 예측 분야 연구 논문 본문 정독 필요 (정확도 수치·방법 비교표 작성).

## 논문 매핑

- **Related Work**: 결과 2의 표를 범주별(양자화 / 오프로딩 / 재계산 / 메모리 예측)로 재구성.
- **Introduction**: "기법은 많지만 조합과 예측은 사용자 몫"이라는 공백 제시.
- 핵심 문장 초안: "기존 메모리 절감 기법은 개별적으로 성숙했으나, 주어진 메모리 예산에서 어떤 기법을 어떻게 조합해야
  하는지는 여전히 사용자의 시행착오에 의존한다."

## 출처 (URL)

- safetensors: https://crates.io/crates/safetensors , https://deepwiki.com/safetensors/safetensors
- bitsandbytes: https://github.com/bitsandbytes-foundation/bitsandbytes
- torchao: https://pytorch.org/blog/pytorch-native-architecture-optimization/
- 양자화 방식 현황: https://www.kunalganglani.com/blog/llm-quantization-levels-q4-q8-fp16
- accelerate 추정기: https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator , https://github.com/huggingface/accelerate/issues/3379
- air-rs: https://pypi.org/project/air-rs/ · oxillama: https://github.com/cool-japan/oxillama
- StreamLoader: https://github.com/madtunebk/StreamLoader
- oLLM: https://www.marktechpost.com/2025/09/29/meet-ollm-a-lightweight-python-library-that-brings-100k-context-llm-inference-to-8-gb-consumer-gpus-via-ssd-offload-no-quantization-required/
- 메모리 예측 연구: https://arxiv.org/abs/2602.17817 , https://arxiv.org/abs/2504.03887 , https://arxiv.org/abs/2512.07853 , https://arxiv.org/abs/2205.12095
- 동명: https://github.com/MeMoPro , https://memopro.ai/
