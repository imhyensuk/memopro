# 0030. 중복성 검사 3차 제안 C1~C5 채택

- **날짜**: 2026-09-25
- **유형**: decision
- **상태**: 확정 — 설계 v0.3.6, 계획 v0.3.5에 반영
- **관련 기록**: 0029 (중복성 검사 3차), 0018 (α 기각), 0021 Q1·Q2, 0022 R2, 0027 (방출 엔진)

## 사용자 결정

> "C1~C5 승인할게."

| # | 결정 | 반영 위치 |
|---|---|---|
| C1 | **α 부정적 결과는 단독 논문으로 쓰지 않는다.** 센서스 논문 Discussion의 한 절로 둔다. Kim et al. 2021(셀프 어텐션 비-Lipschitz), i-ResNet, No Free Swap을 인용한다. 기여는 두 가지로 한정한다. ① 사전학습 모델 잔차 가지의 실측 비수축성 ② 힌트 교정이 오차를 키우는 체인 누적 | 0018 상태 줄(0029에서 표시), `docs/research/README.md` 연구 질문 절 주석 |
| C2 | **E009 기준선**: macOS는 `torch.save`·`numpy.tofile`, Linux는 TensorNVMe를 기준으로 삼는다. 방출 엔진은 신규성을 주장하지 않는다. 존재 이유(macOS 지원, RS4 메모리 상한, β 결합)를 명시한다. Linux에서 TensorNVMe를 선택 백엔드로 쓸지는 E009 결과로 판단한다 | architecture.md §6 `spill` 행, development-plan.md A1 E009 행 |
| C3 | **census 정밀 모드 재정의**: 추론 층별 양자화 민감도(AIMET QuantAnalyzer·HAWQ 등 기존 도구 영역, 재구현하지 않음)가 아니다. **학습 상태 범주별로 필요한 비트 대비 실제 저장 비트**를 측정하는 메모리 조사이다 | architecture.md census 절 |
| C4 | **기존 도구와의 비교를 명시**한다. README에 "기존 도구와의 관계" 표(check·train_session·방출 엔진·census 정밀 모드)를 넣는다. A2 완료 조건에 "같은 과제에서 vram-check·accelerate estimate-memory·ProTrain·AutoCheckpoint·HF `auto_find_batch_size`와 비교하고, 우위가 없으면 문서에 적는다"를 추가한다 | README.md §3, development-plan.md A2 |
| C5 | **v0.3(N3) 착수 조건**: `memopro run`과 γ의 중복성 재조사를 먼저 기록한다 | development-plan.md N3 |

## 남은 결정과의 관계

- **C1과 0021 Q2**: C1은 α 결과를 "센서스 논문의 한 절"로 정했다. 따라서 Q2(논문 주력)의 α 처리 부분은 이 결정으로 확정된다. 다만 "센서스 연구를 주력으로 한다"는 Q2 본안은 **별도 승인이 남아 있다.** 연구 질문(RQ) 갱신도 Q2 결정 후에 한다.
- **C3과 0021 Q1·0022 R2**: C3은 정밀 모드의 **내용**을 정했다. 정밀 모드를 v0.1에 넣을지(Q1)와 census 기능 전체의 재정의(R2)는 여전히 대기 중이다.
- **C2와 0027**: RS1~RS5 규칙은 그대로이다. C2는 엔진의 **평가 기준과 주장 범위**만 정한다.

## 반영한 문서

| 문서 | 변경 |
|---|---|
| `docs/design/architecture.md` | v0.3.6. census 정밀 모드 정의와 "양자화 민감도 분석이 아님" 명시. `spill` 행에 신규성 없음·존재 이유·TensorNVMe 판단 시점 |
| `docs/design/development-plan.md` | v0.3.5. E009 기준선, A2 기존 도구 비교 조건, N3 착수 조건 |
| `README.md` | §3에 "기존 도구와의 관계" 표 |
| `docs/research/README.md` | 0029 상태, 0030 색인, 연구 질문 절에 α 처리 주석 |
| `docs/research/0029` | 상태 줄(본문 보존) |
| `CLAUDE.md` | 고유 영역과 중복 영역, E009 기준선, α 처리 요약 |

## 한계

- 비교 대상 도구 중 일부(TensorNVMe, ProTrain)는 Linux·CUDA 환경이 필요하다. 이 기기(M1)에서는 macOS 기준선만 측정할 수 있다. Linux 비교는 외부 환경이 필요하다.

## 논문 매핑

- **Related Work**: C4의 비교 표를 그대로 쓸 수 있다.
- **Discussion**: C1. "선행 이론 누락 → 사전 등록 실험으로 확인 → 주장 축소"라는 연구 과정을 투명하게 보고한다.
- **Evaluation**: C2·C4는 "기존 도구 대비"라는 평가 축을 만든다. 도구 논문 심사에서 요구되는 비교 기준선이다.
