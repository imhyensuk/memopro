# 0006. 신규 메모리 절감 기법 탐색 및 신규성 검증

- **날짜**: 2026-09-24
- **유형**: survey + design
- **상태**: 확정 (탐색) / 최종 채택 기법은 사용자 승인 대기
- **관련 기록**: 0005

## 배경 / 동기

0005의 조건: G1–G3를 **기존 도구와 겹치지 않는 창의적 방법**으로 달성할 수 있는가.

## 방법

1. 목표별(G1 전 단계 / G2 극한 효율 / G3 하드웨어 병목)로 아이디어 12개를 발산.
2. 각 아이디어의 핵심 메커니즘을 키워드로 웹·arXiv 검색하여 선행 사례 확인 (2026-09 기준).
3. 판정: **중복 높음**(동일 메커니즘 존재) / **중복 중간**(유사 메커니즘이 다른 설정에서 존재) /
   **조사 범위 내 미발견**(핵심 메커니즘 조합을 찾지 못함).
4. 미발견·중간 후보를 개발 난이도, 학술적 가치, 실용성, 절감 효과, 중복성으로 평가.

> **주의**: "미발견"은 "신규함"과 같지 않다. 이번 조사는 웹 검색 수준이며, 논문화 전에 체계적 문헌 조사
> (Google Scholar, Semantic Scholar 인용 추적, MLSys/ICML/NeurIPS/ISCA 목록)가 필요하다.

## 결과 1 — 아이디어 12개 신규성 판정

| # | 아이디어 | 핵심 메커니즘 | 선행 사례 | 판정 |
|---|---|---|---|---|
| 1 | 비트플레인 탄력 정밀도 | 가중치를 비트플레인으로 저장, 메모리 부족 시 하위 비트 제거 | Any-Precision LLM [@park2024anyprecision], MatQuant [@nair2025matquant], **PagedWeight(2026-07, GPU MoE 서빙에서 KV 캐시 크기에 따라 비트폭 동적 조절)** [@yang2026pagedweight], SliceMoE | 🔴 높음 |
| 2 | 메모리 압박 적응 학습 | 가용 메모리에 따라 마이크로배치 실시간 조절 | Tri-Accel(VRAM 가용량 기반 배치 조절) [@triaccel2025], eLLM, 탄력적 분산 학습 다수 | 🟡 중간 |
| 3 | **잔차 고정점 체크포인팅** | 저비트 체크포인트를 잔차 방정식으로 교정하여 복원 | RevNet·Reformer(구조 변경 필요), i-ResNet(고정점 역변환, 립시츠 제약 학습 필요), MEFT·Dr²Net(사전학습 모델을 가역 구조로 변환 + 파인튜닝), ActNN·GACT(양자화 저장, 교정 없음) | 🟢 **미발견** (조합) |
| 4 | **유휴 텐서 동면** | 대화형 개발 세션에서 안 쓰는 텐서를 자동 압축·복원 | 코덱(ZipNN, NeuZip, DFloat11), 저장 포맷(compressed-tensors), OS 범용 메모리 압축 | 🟢 **미발견** (적용 설정) |
| 5 | OS 압축기 친화 레이아웃 | 텐서 바이트를 재배열해 macOS(WKdm/LZ4)·Linux zswap이 더 잘 압축하게 함 | blosc 바이트 셔플(범용 압축), TRACE(CXL 링크 압축) | 🟢 미발견, 효과 불확실 → #4의 실험 변형으로 흡수 |
| 6 | 원소 단위 파라미터 냉각 | 수렴한 파라미터를 동결하고 옵티마이저 상태 해제·압축 | LISA, BAdam, AdaGradSelect, TimelyFreeze(블록·레이어 단위), 희소 파인튜닝(동적 인덱스) | 🟡 중간 |
| 7 | 시드 재생성 텐서 | dropout 마스크 등 난수 텐서를 시드로 재생성 | FlashAttention, torch.compile RNG 재생, MeZO, SeedLM | 🔴 높음 |
| 8 | 무손실 가중치 압축 | 지수부 엔트로피 부호화 | ZipNN [@zipnn2024], NeuZip [@neuzip2024], DFloat11, ZipServ | 🔴 매우 높음 |
| 9 | 레이어 간 델타 부호화 | 잔차 스트림의 층간 차이만 저비트 저장 | (직접 사례 미발견) | 단독 이득 작음 → #3에 흡수 |
| 10 | 점진 정밀도 행렬곱 | 상위 비트부터 계산, 필요할 때만 하위 비트 로드 | MoBiQuant, FlexQuant, Any-Precision 계열 | 🔴 높음 |
| 11 | 프리픽스 공유 학습 | 공통 프롬프트 활성값 공유 | Hydragen, GRPO 프리픽스 그룹화 | 🔴 높음 |
| 12 | fork COW 기반 실험 병렬화 | 하이퍼파라미터 탐색 프로세스들이 동결 가중치를 copy-on-write로 공유 | OS 표준 기법, Ray 공유 객체 저장소, 멀티 LoRA 서빙 | 🟡 중간, 학술 가치 낮음 |

## 결과 2 — 유망 후보 상세

### 후보 α: 잔차 고정점 체크포인팅 (Residual Fixed-point Checkpointing, RFC)

**관찰**: 잔차 네트워크(Transformer, ResNet)의 블록은 `x_{l+1} = x_l + f_l(x_l)`이다.
역전파 때 `x_{l+1}`을 알고 있다면, `x_l`은 방정식 `x = x_{l+1} − f_l(x)`의 **고정점**이다.

**메커니즘**
1. 순전파: 마지막 `x_L`만 원래 정밀도로 저장한다. 각 층 입력 `x_l`은 **2~4비트 힌트** `h_l = Q(x_l)`로만 저장한다.
2. 역전파: `x ← dequant(h_l)`에서 시작해 `x ← x_{l+1} − f_l(x)`를 k회 반복한다(필요 시 Anderson 가속).
3. **자체 검증**: 잔차 `r = ‖x + f_l(x) − x_{l+1}‖`를 원본 없이 계산할 수 있으므로 복원 오차를 **측정**할 수 있다.
   r이 임계값을 넘는 층은 다음 스텝부터 원래 정밀도 체크포인트로 전환한다(적응형).
4. 마지막 반복에서 계산한 `f_l(x)`의 내부 값을 그대로 역전파에 재사용한다. 일반 체크포인팅의 재계산 1회를 대체한다.

**기존 기법과의 차이**
- RevNet/Reformer/MEFT: **모델 구조를 바꿔야** 한다. RFC는 **사전학습 모델을 수정 없이** 사용한다.
- i-ResNet: 수렴을 보장하려고 립시츠 상수를 1 미만으로 제약해 학습한다. RFC는 제약 없이 **저비트 힌트로 시작점을 해 근처에 두어** 국소 수렴을 노린다.
- ActNN/GACT: 양자화 오차를 그대로 받아들인다. RFC는 **네트워크 자신의 방정식으로 오차를 교정**한다.

**가설 (검증 필요)**
- H1: Pre-LN Transformer에서는 잔차 스트림 노름이 깊이에 따라 커지므로, LayerNorm을 거친 `f_l`의 야코비안 노름이 대부분 층에서 1 미만이다 → 국소 수렴.
- H2: 4비트 힌트에서 시작하면 k ≤ 3회 반복으로 bf16 수준 오차에 도달한다.
- H3: 복원된 활성값으로 계산한 그래디언트의 코사인 유사도가 정확한 그래디언트 대비 0.999 이상이다.

**예상 효과 (가설 기반 추정)**
- 체크포인트 메모리: 층별 bf16(16비트) → 4비트 힌트면 약 **4배**, 2비트면 약 **8배** 절감 (`x_L` 1개는 원래 정밀도로 저장).
- 연산 비용: 일반 체크포인팅(재계산 1회) 대비 층당 순전파 (k−1)회 추가. k=3이면 학습 스텝이 약 +50%.
- G3와의 연결: 연산량(FLOPs)은 메모리 용량·대역폭보다 빠르게 늘어난다(memory wall). RFC는 **메모리 요구를 연산으로 치환**한다.

**위험**: 초반 층·어텐션에서 수렴하지 않을 수 있다(→ 적응형 폴백). 층을 거꾸로 복원하면서 오차가 누적될 수 있다. dropout 난수는 같은 시드로 재생해야 한다.

### 후보 β: 유휴 텐서 동면 (Idle Tensor Hibernation)

**관찰**: 개발 단계(G1)의 Jupyter·REPL 세션에서는 변수가 스코프를 벗어나지 않아, 이미 쓰지 않는 텐서(이전 실험의 모델, 중간 결과)가
커널을 재시작할 때까지 메모리를 차지한다. JupyterLab에는 이를 정리하는 기본 기능이 없다.
8GB 기기에서는 이것이 OOM의 흔한 원인이다.

**메커니즘**
1. 사용자 네임스페이스의 텐서를 추적한다(IPython 셀 실행 훅 + `TorchDispatchMode`로 접근 감지).
2. N개 셀·T초 동안 접근되지 않은 텐서는 **동면**시킨다. Rust 코덱으로 압축하고 원래 저장공간은 해제한다.
   - 무손실: 바이트플레인 분리 + 엔트로피 부호화 (bf16 약 1.5배, fp32 약 1.2배 — ZipNN 계열 보고치)
   - 선택형 손실: fp32 → bf16 등 (약 2배)
   - 추가 계층: SSD로 방출 (RAM 절감 약 100%)
3. 동면한 텐서에 연산이 닿으면 dispatch 단계에서 가로채 **투명하게 복원**한다. 사용자 코드는 바꾸지 않는다.
4. 실험 변형(#5): 직접 압축하는 대신 바이트만 재배열해 두고, OS 메모리 압축기(macOS WKdm/LZ4)가 압축하게 둔다.

**기존 기법과의 차이**: 코덱 자체(#8)는 기존 기술이다. 새로운 점은 **대화형 개발 세션에서 텐서 생애 주기를 추적해 자동으로 동면·복원**하는 적용 설정이다.

**예상 효과**: 유휴 텐서 비율 × 압축률만큼 줄어든다. SSD 방출을 쓰면 유휴 텐서의 RAM 점유가 거의 0이 된다. 학술 가치는 낮고 실용성은 높다.

### 후보 γ: OS 메모리 압박 기반 탄력 런타임 (Elastic Runtime)

**메커니즘**: OS의 메모리 압박 신호(macOS dispatch memory pressure / Linux PSI·cgroup)를 Rust 코어가 구독한다.
압박이 커지면 단계적으로 대응한다.
- 추론: 비트플레인 하위 비트를 제거한다(16 → 8 → 4비트).
- 학습: 마이크로배치를 줄이고 그래디언트 누적을 늘린다.
- 개발: β의 동면을 가속한다.
압박이 풀리면 복구한다. 목표는 **OOM이나 스왑 폭주로 죽지 않고 우아하게 열화하는 것(graceful degradation)**이다.

**기존과의 차이**: PagedWeight는 GPU 서빙에서 KV 캐시와의 균형을, Tri-Accel은 VRAM 가용량 기반 배치 조절을 다룬다.
γ는 **OS 전역 압박 신호, 통합 메모리 기기, 다른 앱과 메모리를 공유하는 개인 기기**를 대상으로 한다. 차별점이 좁고 중복은 중간 수준이다.

## 결과 3 — 후보 평가 (5점 척도, 중복성은 적을수록 높음)

| 평가 요소 | α 잔차 고정점 체크포인팅 | β 유휴 텐서 동면 | γ 탄력 런타임 |
|---|---|---|---|
| 개발 난이도 (쉬울수록 ↑) | 2 — autograd 통합, 수렴 제어 | 3 — dispatch 가로채기, 코덱 | 2.5 — OS별 신호, 다중 정책 |
| 학술적 가치 | **4.5** — 새 메커니즘, 이론(수렴 조건)과 실험 모두 가능 | 1.5 — 시스템 도구 | 2.5 — 시스템 논문감, 차별점 좁음 |
| 실용성 | 3.5 — 학습 시 활성값 메모리가 주 병목인 경우 | **4.5** — 모든 노트북 사용자, 즉시 체감 | 3.5 — 개인 기기 사용자 |
| 메모리 절감 효과 | **4** — 체크포인트 4~8배 | 3 — 유휴 비율에 좌우 | 3 — 상황 적응형 |
| 중복성 (적을수록 ↑) | **4.5** | 4 | 2.5 |
| 목표 적합 G1 / G2 / G3 | 중 / **강** / 중(메모리 → 연산 치환) | **강**(개발 단계) / 중 / 약 | 중 / 강 / 중 |
| **합계 (25점)** | **18.5** | **16** | 14 |

참고: 0004의 플래너 C'는 16.5, 기존 기법 재구현 A는 11.5.

## 결론 및 권고

1. **주력(학술): α 잔차 고정점 체크포인팅.** 논문의 핵심 기여 후보이다. 먼저 **가설 H1~H3 검증 실험**을 해야 하며, 실패하면 폐기하거나 수정한다.
2. **보조(실용): β 유휴 텐서 동면.** 사용자가 바로 체감할 수 있는 첫 배포 기능이다. Rust 코덱이 α의 힌트 저장과 γ에서 재사용하는 공통 기반이 된다.
3. **γ는 후순위.** α와 β의 기반(코덱, 비트플레인, 텐서 원장)이 갖춰진 뒤 확장한다.

공통 Rust 코어: 부동소수 인식 코덱(바이트플레인 분리 + 엔트로피 부호화 + 저비트 양자화), 텐서 원장(생애 주기·온도 추적), OS 메모리 신호 모니터.

## 다음 실험 계획 (0007 예정)

- 목적: α의 H1~H3를 실제 사전학습 모델로 검증.
- 모델: GPT-2 small(124M, Pre-LN). M1 8GB에서 실행할 수 있다.
- 측정: 층별 야코비안 스펙트럴 노름 추정(멱반복법), 힌트 비트수(2/3/4) × 반복 횟수(k=1~5)별 복원 오차, 그래디언트 코사인 유사도, 소규모 파인튜닝 손실 곡선.
- 필요 환경: 가상환경 + PyTorch + transformers + GPT-2 가중치(약 0.5GB 다운로드).

## 한계

- 신규성 판정은 웹 검색 기반이다. 체계적 문헌 조사가 필요하다. 특히 α는 "fixed-point", "Anderson acceleration", "implicit inversion", "activation reconstruction" 키워드로 추가 확인해야 한다.
- 예상 효과 수치는 가설 기반 추정이며 실험 전이다.

## 논문 매핑

- **Introduction**: "메모리를 연산으로 치환" 관점(memory wall), 구조 변경 없는 사전학습 모델 대상.
- **Related Work**: 결과 1 표(가역 네트워크 / 활성값 압축 / 체크포인팅 / 가변 정밀도).
- **Method**: α 메커니즘, 수렴 조건(H1), 자체 검증 잔차와 적응형 폴백.
- **Experiments**: 0007 계획.
- 핵심 문장 초안: "잔차 블록의 입력은 출력과 블록 함수로 정의되는 고정점 방정식의 해이다. 본 연구는 이 성질을 이용해
  저비트로 저장한 활성값을 역전파 시점에 네트워크 자신의 방정식으로 교정하며, 모델 구조 변경 없이 체크포인트 메모리를 줄인다."

## 출처 (URL)

- Any-Precision LLM: https://arxiv.org/abs/2402.10517 · AnyBCQ: https://arxiv.org/abs/2510.10467 · MoBiQuant: https://arxiv.org/abs/2602.20191 · FlexQuant: https://arxiv.org/abs/2506.12024
- MatQuant: https://arxiv.org/abs/2502.06786 · SliceMoE: https://arxiv.org/abs/2512.12990
- PagedWeight: https://arxiv.org/abs/2607.16184
- RevNet: https://arxiv.org/abs/1707.04585 · i-ResNet: https://proceedings.mlr.press/v97/behrmann19a/behrmann19a.pdf · Reformer: https://arxiv.org/abs/2001.04451
- Dr²Net: https://arxiv.org/abs/2401.04105 · 가역 설계(CNN): https://arxiv.org/abs/1910.11127 · 가역 Transformer 블록 개관: https://www.emergentmind.com/topics/reversible-transformer-block-architectures
- GACT: https://arxiv.org/abs/2206.11357 · AC-GC: https://proceedings.neurips.cc/paper/2021/file/e655c7716a4b3ea67f48c6322fc42ed6-Paper.pdf · COMET: https://arxiv.org/abs/2111.09562
- StreamBP: https://arxiv.org/abs/2506.03077
- Tri-Accel: https://arxiv.org/abs/2508.16905 · eLLM: https://arxiv.org/abs/2506.15155
- AdaGradSelect: https://arxiv.org/abs/2512.15764 · TimelyFreeze: https://arxiv.org/abs/2602.05754
- ZipNN: https://arxiv.org/abs/2411.05239 · NeuZip: https://www.researchgate.net/publication/385317689 · ZipServ: https://arxiv.org/abs/2603.17435 · 무손실 압축(저정밀 포맷): https://arxiv.org/abs/2508.19263
- WKdm·macOS 압축: https://lwn.net/Articles/571898/ , https://www.researchgate.net/publication/264086271 · TRACE: https://arxiv.org/abs/2509.03377
- Jupyter 메모리 문제: https://pub.towardsai.net/your-jupyterlab-is-hoarding-dead-sessions-heres-how-i-fixed-it-a7543bba6def , https://fastai1.fast.ai/tutorial.resources.html
- compressed-tensors: https://github.com/vllm-project/compressed-tensors
