# memopro 연구 기록 (Research Log)

향후 **연구보고서·논문** 작성을 위한 1차 자료. 모든 개발 단계마다 기록을 남긴다.

## 기록 원칙

1. **단계마다 기록한다.** 조사, 설계 결정, 구현, 실험, 실패까지 모두. 실패와 철회한 결정도 논문의 근거가 된다.
2. **번호 + 날짜.** 파일명 `NNNN-짧은-제목.md`, 본문 상단에 날짜(YYYY-MM-DD). 번호는 시간순, 재사용하지 않는다.
3. **기존 기록은 수정하지 않는다.** 결정이 바뀌면 새 기록을 추가하고 이전 기록 상태를 `대체됨(→ NNNN)`으로 표시한다.
4. **재현 가능하게.** 실험은 환경(하드웨어, OS, 버전), 명령어, 입력, 원시 결과 파일 위치를 남긴다.
5. **출처를 남긴다.** 인용할 문헌·도구는 [references.bib](references.bib)에 추가하고 본문에서 `[@key]`로 참조한다.
6. **논문 매핑.** 각 기록 끝에 이 내용이 논문의 어느 절(Introduction / Related Work / Method / Experiments / Discussion)에 쓰일지 적는다.

## 기록 유형

| 유형 | 용도 | 논문에서 |
|---|---|---|
| `survey` | 선행 연구·도구 조사 | Related Work |
| `decision` | 설계·방향 결정 (ADR) | Method, Discussion |
| `design` | 알고리즘·아키텍처 설계 | Method |
| `experiment` | 실험 설계·결과 | Experiments |
| `milestone` | 릴리스·단계 완료 요약 | 전체 흐름 |

새 기록은 [_template.md](_template.md)를 복사해 작성한다.

## 색인

| # | 날짜 | 유형 | 제목 | 상태 |
|---|---|---|---|---|
| [0001](0001-project-inception.md) | 2026-09-24 | milestone | 프로젝트 착수: 비전과 목표 | 확정 |
| [0002](0002-prior-art-survey.md) | 2026-09-24 | survey | 선행 기술 조사 및 중복성 검사 | 일부 정정(→0004) |
| [0003](0003-positioning-decision.md) | 2026-09-24 | decision | 포지셔닝 전환: 기법 구현 → 메모리 플래너 | 대체됨(→0005) |
| [0004](0004-planner-evaluation.md) | 2026-09-24 | survey+decision | 개선안 다면 평가 및 선행 조사 정정 | 평가 확정, 권고안 기각(→0005) |
| [0005](0005-direction-technique-reimplementation.md) | 2026-09-24 | decision | 방향 결정: 개별 기법 구현 노선 (신규성 조건) | 확정 |
| [0006](0006-novel-technique-exploration.md) | 2026-09-24 | survey+design | 신규 메모리 절감 기법 탐색 및 신규성 검증 | 탐색 확정, 채택 승인 대기 |
| [0007](0007-radical-novelty-assessment.md) | 2026-09-24 | survey+decision | 완전한 신규성 가능성 검토, 관찰 우선 방법론 제안 | 검토 확정, 방법론 승인 대기 |
| [0008](0008-library-design-and-stages.md) | 2026-09-24 | design+decision | 라이브러리 설계 및 개발 단계 구조화 | 설계 v0.1 → 대체됨(→0010) |
| [0009](0009-use-case-analysis.md) | 2026-09-24 | design | 활용 환경 분석: LoRA 보완 관계, 추론 단계 공백 | 확정, 추론 공백 해소(→0010) |
| [0010](0010-universality-redesign.md) | 2026-09-24 | decision+design | 범용성 중심 재설계: 범용 접근 계층 + 연구 코어 | 설계 v0.2 확정, 논리 수정(→0011) |
| [0011](0011-design-v02-verification.md) | 2026-09-25 | survey+decision | 설계 v0.2 검증: 중복성(D1~D7)·논리성(L1~L14) | 검증 확정, S1·S3 승인 / S2 기각(→0012) |
| [0012](0012-revision-v03.md) | 2026-09-25 | decision | 수정안 v0.3: 대상 재정의(S1), v0.1 재구성(S3), 외부 안내 기각(S2) | 확정, 검증(→0013) |
| [0013](0013-revision-v03-verification.md) | 2026-09-25 | survey+decision | 수정안 v0.3 검증: 중복성(D8~D13)·논리성(V1~V15) → 설계 v0.3.1 | 확정 |
| [0014](0014-self-critique-v031.md) | 2026-09-25 | survey+decision | 설계 v0.3.1 자기 비판: 개선 제안 P1~P10 | P1~P4 채택(→0015), P5~P10 보류 |
| [0015](0015-adopt-p1-p4.md) | 2026-09-25 | decision | P1~P4 채택: 실험 우선, 범위 축소·걷는 뼈대, OS 기준선, β 명시 실행 → 설계 v0.3.2 | 확정 |
| [0016](0016-s1-foundation.md) | 2026-09-25 | milestone | S1 기반 구축과 걷는 뼈대 (빌드·wheel·sdist·crate 패키징 관통, 미배포) | 확정 |
| [0017](0017-x1-preregistration.md) | 2026-09-25 | experiment | X1 첫 실험 사전 등록: α E001~E003 판정 기준, E005 β 설계 판정 규칙, E006 검증 기준 | 확정 (실행 전 작성) |
| [0018](0018-alpha-e001-e003-results.md) | 2026-09-25 | experiment | α 검증 E001~E003 결과: H1·H2·H3 기각, Gα 중간 판정 = 기각 (부정적 결과) | 확정 |
| [0019](0019-e005-redundancy-results.md) | 2026-09-25 | experiment | E005: 학습 텐서의 무손실 중복 ≈14%, OS 방식 압축은 fp32에 무효 → β 규칙 2 발동 | 확정 |
| [0020](0020-e006-idle-probe.md) | 2026-09-25 | experiment | E006: 노트북 유휴 메모리 계측 도구, 합성 세션 검증 통과 | 확정 (실데이터 미수집) |
| [0021](0021-x1-consequences.md) | 2026-09-25 | decision | X1 결과에 따른 개정: α 트랙 종료, β 기본 모드 변경, 로드맵 재번호 / 제안 Q1~Q3 | D1~D3 확정, Q1~Q3 승인 대기 |
| [0022](0022-design-impact-synthesis.md) | 2026-09-25 | survey+decision | 종합: 조사·검증·실험이 초기 설계에 준 영향, 설계 규칙 K1~K6 / 제안 R1~R5 | 종합 확정, R1~R5 승인 대기 |
| [0023](0023-e007-e008-preregistration.md) | 2026-09-25 | experiment | 사전 등록: E007 Muon 타당성, E008 Rust 병렬 방출 코덱 | 확정 (실행 전 작성) |
| [0024](0024-e007-muon-results.md) | 2026-09-25 | experiment | E007: Muon — 메모리 −34%·처음부터 학습 품질 우위, 그러나 M1 소배치에서 3배 느림 → 후보 미추가 | 확정 |
| [0025](0025-e008-rust-results-and-strategy.md) | 2026-09-25 | experiment+design+decision | E008: 단순 Rust는 Python 스레드보다 느림, 압축 방출은 무압축보다 느림 → β 방출 기본 무압축 / E008b(탐색): 설계한 Rust는 1.49배·방출 −18% / 전략 RS1~RS8 | 판정 확정, RS1~RS5 채택·RS6~RS8 보류(→0027) |
| [0026](0026-min-python-311.md) | 2026-09-25 | decision | 최소 Python 3.10 → 3.11 상향(abi3-py311, buffer protocol 전제)과 3.10 흔적 정리 | 확정 |
| [0027](0027-adopt-rs1-rs5.md) | 2026-09-25 | decision | Rust 코어 설계 규칙 RS1~RS5 채택(완료 조건 확정), RS6~RS8 보류 → A1 = 방출 엔진, 방출 엔진 → E009 순서 | 확정 |

## 연구 질문 (Research Questions) — 초안 (0006 기준으로 갱신)

기록이 쌓이면서 갱신한다. 이전 버전(플래너 노선, RQ1–RQ3)은 0003/0004 참조.

- **RQ1 (수렴성)** 사전학습된 Pre-LN Transformer의 잔차 블록에서, 저비트 힌트로 시작한 고정점 반복이 입력 활성값으로 수렴하는가? 수렴 조건은 무엇인가?
- **RQ2 (정확도-메모리-연산 트레이드오프)** 힌트 비트수와 반복 횟수에 따라 체크포인트 메모리, 그래디언트 오차, 학습 시간이 어떻게 달라지는가?
  기존 체크포인팅, 활성값 양자화(ActNN/GACT), 가역 변환(MEFT) 대비 파레토 우위가 있는가?
- **RQ3 (학습 품질)** 복원된 활성값으로 학습해도 최종 손실과 다운스트림 정확도가 유지되는가?
- **RQ4 (개발 단계 메모리)** 대화형 세션에서 유휴 텐서가 차지하는 메모리 비율은 얼마이며, 자동 동면으로 얼마나 회수할 수 있는가?

## 평가 지표 후보

- 체크포인트·활성값 피크 메모리 (bytes), 절감 배수
- 복원 오차 (상대 L2), 그래디언트 코사인 유사도
- 학습 스텝 시간 오버헤드 (%), 처리량 (tokens/s)
- 최종 손실·다운스트림 정확도 변화
- 동면: 회수 메모리, 복원 지연(ms), 압축률
