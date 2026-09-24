# 0004. 개선안(메모리 플래너) 다면 평가 및 선행 조사 정정

- **날짜**: 2026-09-24
- **유형**: survey + decision
- **상태**: 확정 (평가) / 권고안 C' 기각(→ 0005)
- **관련 기록**: 0002 (일부 정정), 0003 (범위 조정 권고)

## 배경 / 동기

0003에서 채택한 "메모리 플래너(estimate → plan → apply)"를 개발 난이도, 학술적 가치, 실용성, 메모리 절감 효과,
중복성(상용·오픈소스 유사 도구) 기준으로 재평가한다. 이를 위해 0002에서 다루지 않은 **"예산 기반 자동 설계" 계열 도구**를 추가 조사했다.

## 추가 조사 결과 — 0002 정정

0002는 "예산 기반 자동 조합: 해당 범용 도구 미발견"이라고 결론지었으나, **단일 축(single-axis) 플래너는 이미 다수 존재**한다.

| 도구 / 연구 | 무엇을 자동화하나 | 대상 하드웨어 필요 여부 | memopro와의 관계 |
|---|---|---|---|
| **PyTorch `torch._functorch.config.activation_memory_budget`** (2.4+) [@pytorch_ac_blog] | 메모리 예산에 맞춰 재계산(recompute) 대상을 min-cut으로 자동 선택 | 컴파일 시점 (실행 환경) | **plan의 재계산 축과 직접 중복.** 단, 활성값만 고려하고 재계산 중 메모리는 무시, 예산 대비 실제 피크가 비단조(non-monotonic)라는 이슈 보고(pytorch#197838), RNG 연산 재계산 시 그래디언트 오염 버그 보고 |
| torchtitan budget-aware activation policy (PR #4636) | 위와 유사, 대규모 사전학습용 | 필요 | 동일 축 |
| **DeepSpeed Autotuning** [@deepspeed_autotuning] | ZeRO 단계·마이크로배치 등을 모델 프로파일링 + 실험으로 탐색. 단계별 최소 메모리 추정 포함 | **필요 (실제 실행)** | plan과 가장 가까운 상용급 도구. 분산 학습 중심, 양자화 미포함 |
| **Colossal-AI Gemini** [@bian2021colossal] | 워밍업 중 메모리 추적 후 GPU↔CPU 텐서 배치를 런타임 자동 관리 | **필요 (런타임)** | 오프로드 축 중복. HPC-AI Tech가 상용화 |
| Checkmate [@jain2020checkmate] | 메모리 예산 하 최적 재계산 스케줄(ILP) | 불필요 (정적) | 학술 선행, 재계산 단일 축 |
| MONeT [@shah2021monet], DTR [@kirisame2021dtr], Mimose [@mimose2022] | 재계산 스케줄 (연산자 선택 포함 / 동적 / 입력 인지) | 다양 | 학술 선행 |
| **POET** [@patil2022poet] | 소형 기기에서 **재계산 + 페이징(오프로드) 공동 최적화** (MILP) | 불필요 (정적) | **두 축 공동 최적화의 가장 가까운 선행 연구** — 반드시 비교·인용 |
| FlexGen [@sheng2023flexgen] | 추론 시 GPU/CPU/디스크 배치 정책을 LP로 탐색 | 부분 | 추론 오프로드 축 |
| HF Trainer `auto_find_batch_size`, Lightning batch size finder, Composer 자동 마이크로배치 | OOM 나면 배치를 줄여 재시도 | **필요** | 경험적 시행착오 방식 |
| **VRAM 계산기** 다수 (llm_vram_calc, vram_estimator, llm-memory-calculator, LLMScale, HF Space 등) | LLM 수식 기반 메모리 계산 | 불필요 | **estimate와 직접 경쟁.** 대부분 Transformer LLM 전용 공식, 그래프 수준 분석 아님 |
| **Unsloth** (상용 회사) | 양자화 + 체크포인팅 + 오프로딩을 묶어 제공 | 필요 | apply의 "묶음 적용"과 유사하나 고정 레시피, 모델군 한정 |

### 여전히 남는 공백 (좁혀진 범위)

1. **대상 하드웨어 없이 사전 설계**: DeepSpeed Autotuning, Gemini, auto_find_batch_size는 모두 목표 GPU에서 실제로 실행해야 한다.
   "8GB 노트북에서, 24GB GPU에 들어갈지와 최적 설정을 미리 안다"는 기능은 비어 있다.
2. **양자화·옵티마이저 정밀도까지 포함한 공동 최적화**: 기존 플래너는 재계산(Checkmate, torch budget) 또는 재계산+페이징(POET)
   또는 분산 설정(DeepSpeed)에 머문다. 가중치 양자화 + 옵티마이저 정밀도 + 재계산 + 오프로드 + 배치를 **한 번에** 다루는 범용 도구는 확인되지 않았다.
3. **그래프 수준의 범용 예측**: 계산기들은 LLM 공식 기반. 임의의 PyTorch 모델(CNN, Diffusion, 멀티모달)에 대한 그래프 수준 피크 예측은 연구 프로토타입 단계.
4. **통합 메모리(Apple Silicon, MPS) 지원**: 조사된 플래너는 거의 CUDA 전제.

## 다면 평가

점수: 5점 척도 (높을수록 유리). 중복성은 "중복이 적을수록 높은 점수".

### 구성요소별

| 평가 요소 | estimate (v0.1) | plan (v0.2) | apply (v0.3) |
|---|---|---|---|
| 개발 난이도 (쉬울수록 ↑) | 3 — 수식 수준은 쉬움, 그래프+liveness+할당자 보정으로 ±5% 달성은 어려움 | 2 — 탐색 공간 설계, **시간 비용 모델**(속도 저하 예측)이 핵심 난제 | 2 — 외부 도구 API 변화 추적, 기법 간 호환성 버그(예: RNG 재계산) 유지보수 부담 |
| 학술적 가치 | 2.5 — 활발한 연구 분야이나 SOTA 대비 개선 입증 필요 | **4** — 다축 공동 최적화 + 하드웨어 비의존 사전 설계는 기여로 주장 가능 | 1.5 — 공학적 통합, 학술 기여 낮음 |
| 실용성 | 4 — 수요 확실(계산기가 많다는 것 자체가 수요 증거) | 4 — 신뢰도가 확보되면 높음 | 4 — 사용자 체감이 가장 큼 |
| 메모리 절감 효과 | 1 (직접) / 2 (간접: 과잉 확보·OOM 방지) | 3 — 기존 기법 상한 내에서 **더 나은 조합 선택** | 3 — plan과 동일 (실제 절감은 하위 도구가 수행) |
| 중복성 (적을수록 ↑) | **1.5** — 계산기 다수, accelerate, DeepSpeed 추정 함수 | 3 — 단일 축 플래너 다수, 다축·사전 설계는 공백 | 2 — Unsloth, HF/accelerate 설정 묶음 |

### 개선안 전체

| 평가 요소 | 점수 | 요약 |
|---|---|---|
| 개발 난이도 | **2 / 5 (어려움)** | 1인 개발 기준 plan까지 상당한 기간 필요. 정확도 검증에 CUDA 환경 필수 |
| 학술적 가치 | **3.5 / 5** | 기여 지점은 plan에 집중. estimate 단독 논문은 약함. POET·Checkmate·torch budget·DeepSpeed Autotuning 대비 우위 실험 필요 |
| 실용성 | **4 / 5** | 수요 명확. 단 예측이 틀려 OOM이 나면 신뢰가 무너지므로 **보수적 추정(과소예측 방지)** 필수 |
| 메모리 절감 효과 | **2.5 / 5** | memopro 자체는 바이트를 줄이지 않는다. 절감 상한 = 기존 기법의 상한. 이득은 "사용자가 고르는 조합 대비 개선분" |
| 중복성 | **3 / 5** | 0002 판단(빈 자리)보다 경쟁이 많음. 차별점은 좁지만 실재함(위 공백 1–4) |
| 목표 적합성 G1 / G2 / G3 | 강 / 중 / **약** | G3(하드웨어 병목 극복)는 비용 모델에 대역폭을 넣는 수준에 그침 |

### 핵심 약점 (솔직한 평가)

1. **메모리를 직접 줄이지 않는다.** "memopro를 쓰면 몇 % 절감"이라는 효과는 전부 하위 도구의 성과로 귀속될 수 있다.
   평가 지표를 "동일 예산에서 속도·배치 크기 개선" 또는 "OOM 없이 도달한 최소 예산"으로 명확히 정의해야 한다.
2. **estimate는 경쟁이 가장 많다.** v0.1을 단순 계산기 수준으로 내면 기존 계산기들과 구별되지 않는다.
   → v0.1부터 **그래프 수준 + 임의 PyTorch 모델 + 대상 하드웨어 지정 예측**을 차별점으로 삼아야 한다.
3. **Rust 코어의 필요성이 약하다.** 수천 노드의 liveness 분석은 Python으로도 충분히 빠르다.
   Rust가 제값을 하는 곳은 plan의 **조합 탐색(ILP/DP/탐색)**이다. crates.io 배포의 명분은 "프레임워크 비의존 IR"과 Rust ML 생태계(candle, burn) 연동.
4. **G3가 약하다.** 하드웨어 병목(대역폭) 극복은 설계만으로는 한계가 있다. 오프로드 시 prefetch 스케줄 생성 등 Phase 4 과제로 명시해야 한다.
5. **검증 환경 제약.** 개발 기기(M1 8GB)로는 CUDA 할당자 동작 검증 불가 → Colab·클라우드 GPU 예산 필요.

### 대안과의 비교 (0003의 선택지 재평가)

| 선택지 | 난이도 | 학술 | 실용 | 절감 효과 | 중복성 | 합계 |
|---|---|---|---|---|---|---|
| A. 개별 기법 재구현 | 3 | 1.5 | 2 | 4 | 1 | 11.5 |
| B. Apple Silicon 전용 커널 | 2 | 2.5 | 3 | 4 | 3 | 14.5 |
| **C. 메모리 플래너 (현안)** | 2 | 3.5 | 4 | 2.5 | 3 | **15** |
| C' = C + B 요소 (권고안) | 2 | 4 | 4 | 3 | 3.5 | **16.5** |

→ **C 유지가 타당**하되, 범위를 좁히고 차별점을 명시한 C'를 권고한다.

## 권고안 (C')

1. **차별점 명시 (문서·패키지 설명)**
   - 하드웨어 비의존 사전 설계: "지금 기기에서, 목표 기기용 메모리 계획을 세운다."
   - 다축 공동 최적화: 양자화 + 옵티마이저 정밀도 + 재계산 + 오프로드 + 배치/그래디언트 누적.
   - 통합 메모리(MPS)를 CUDA와 동등한 1급 대상으로 지원.
2. **v0.1 estimate 범위 재정의**: LLM 공식 계산기가 아니라 **그래프 수준(FakeTensor/meta 추적) 예측 + 대상 디바이스 프로파일**. 과소예측 방지를 위한 보수적 상한 제공.
3. **비교 기준선(baseline) 확정** — 논문 실험의 비교 대상:
   - estimate: `accelerate estimate-memory`, 수식 계산기, (연구) 동적 분석 기반 예측
   - plan: `torch activation_memory_budget`, DeepSpeed Autotuning(단일 GPU 설정), POET(가능 시), 수동 설정, `auto_find_batch_size`
4. **Rust 역할 재정의**: IR + liveness + **planner 탐색**을 Rust에 둔다. estimate 단계에서 Rust 도입이 과하다고 판단되면 Python으로 먼저 검증 후 이식하는 경로도 허용.
5. **G3 보강 과제 명시**: Phase 4에 대역폭 인지 오프로드·prefetch 스케줄 생성 추가.

## 한계

- 점수는 조사 기반 정성 평가이며, 정량 근거는 Phase 1 실험 이후 갱신한다.
- Unsloth 등 상용 제품의 내부 동작은 공개 자료 범위에서만 판단.
- Mimose 등 일부 문헌 서지 정보 미확인 (references.bib TODO).

## 논문 매핑

- **Related Work**: 위 "추가 조사 결과" 표를 재계산 / 오프로드 / 분산 튜닝 / 예측 / 계산기 범주로 재구성. POET·Checkmate·torch budget·DeepSpeed Autotuning은 필수 인용.
- **Introduction (Contribution)**: 공백 1–4를 기여 목록으로 전환.
- **Experiments**: 권고안 3의 기준선 목록.
- **Discussion / Limitations**: 핵심 약점 1(직접 절감 아님), 4(G3 한계).
- 핵심 문장 초안: "기존 메모리 플래너는 대상 하드웨어에서의 실행을 전제하거나 단일 기법 축에 한정된다. 본 연구는 실행 없이,
  대상 하드웨어 없이, 양자화·재계산·오프로드·배치 구성을 공동으로 최적화하는 사전 메모리 설계를 제안한다."

## 출처 (URL)

- PyTorch activation checkpointing 블로그: https://pytorch.org/blog/activation-checkpointing-techniques/
- pytorch 이슈: https://github.com/pytorch/pytorch/issues/197838 , https://github.com/pytorch/pytorch/issues/161650
- RNG 재계산 버그 가드: https://github.com/zhuhroscar-tech/torch-memory-budget-rng-guard
- torchtitan PR: https://github.com/pytorch/torchtitan/pull/4636
- autocheckpoint: https://github.com/archakamk/autocheckpoint
- DeepSpeed Autotuning: https://www.deepspeed.ai/tutorials/autotuning/
- Colossal-AI Gemini: https://colossalai.org/docs/advanced_tutorials/meet_gemini/ , https://arxiv.org/abs/2110.14883
- Mimose: https://arxiv.org/abs/2209.02478 · ActNN: https://arxiv.org/abs/2104.14129 · AGoQ: https://arxiv.org/abs/2605.00539
- MemAscend: https://arxiv.org/abs/2505.23254 · FlashOptim: https://arxiv.org/abs/2602.23349
- VRAM 계산기: https://github.com/SaehwanPark/llm_vram_calc , https://github.com/mjk0618/vram_estimator , https://github.com/AndreaPi/llm-memory-calculator , https://github.com/tahircengiz/LLMScale
