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
| [0028](0028-status-and-roadmap.md) | 2026-09-25 | survey+design | 현황 종합: 성과, 연구 결과(E001~E008b), 최종 설계(v0.3.5), 남은 개발 단계와 결정 대기 항목 | 확정 (현황) |
| [0029](0029-redundancy-check-v035.md) | 2026-09-25 | survey | 중복성 검사 3차: 최종 설계 v0.3.5 기능(D14~D24)과 연구 주장(D25~D29). 고유 영역 β·census·γ 유지, check·train_session 중복 상승, α 부정적 결과는 기존 이론으로 예측 가능 / 제안 C1~C5 | 조사 확정, C1~C5 승인(→0030) |
| [0030](0030-adopt-c1-c5.md) | 2026-09-25 | decision | 제안 C1~C5 채택: α 부정적 결과는 센서스 논문 Discussion 한 절, E009 기준선(TensorNVMe 등), census 정밀 모드 재정의, A2 기존 도구 비교, v0.3 착수 전 재조사 → 설계 v0.3.6·계획 v0.3.5 | 확정 |
| [0031](0031-adoption-self-critique.md) | 2026-09-25 | survey+decision | 자기 비판 2차(개발자 채택 관점): 대부분 미채택 가능성, β는 좁은 사용자층에 유효 / 개선 방안 I1~I11(수요 측정 Gβ 우선), 대기 제안 H1~H6 기록 | 평가 확정, I1~I5·I7~I9·H1~H6 채택(→0032), I6·I10·I11 대기 |
| [0032](0032-adopt-group1-2.md) | 2026-09-25 | decision | 개발 착수 전 결정(I1·R1·H1~H4·H6·P5·I4·I5·병합)과 v0.1 범위(Q1·R2·P6·P7·R3·I2·I3·I7·I8·I9) 일괄 채택 → Gβ 관문, 순서 변경, β SSD 정책, 제품 중심 β + census | 확정 |
| [0033](0033-research-direction.md) | 2026-09-25 | decision | 연구 방향 확정: 논문 A(센서스 측정 연구) 주력·E012, 논문 B는 v0.1 이후, G3 장기 과제와 후보 탐색 절차, K1~K6 원칙화, 연구·제품 성공 기준 분리 | 확정 (게재처·GPU 환경 대기) |
| [0034](0034-library-skeleton.md) | 2026-09-25 | design+milestone | S2 라이브러리 전체 뼈대: 공개 API(지연 import), 오류 계층(NotYetImplemented), 설정 계층, β 방법 선택 정책, fail-open, CLI 종료 코드, 노트북 매직, Rust 모듈(error·hwinfo·spill·ledger·pressure). 테스트 Python 60·Rust 17 | 확정 |
| [0035](0035-hwinfo-doctor.md) | 2026-09-25 | design+milestone | A1a hwinfo·N1a doctor: 보수적 가용 메모리(macOS 1.55 대 OS 추정 3.27 GiB 발견), 풀별 예산 규칙, `memopro doctor`. macOS 검증, Linux 컨테이너는 CI 대기 | 확정 (Linux 미검증) |
| [0036](0036-build-v01-end-to-end.md) | 2026-09-25 | decision | v0.1 전 구간 제작 결정: Gβ를 배포 전 검증으로 이동, 구현 설계 B1(정체성 유지 해제)~B5 | 확정 |
| [0037](0037-census-v01.md) | 2026-09-25 | design+milestone | N1b census: 범주별 바이트·엔트로피·필요 비트(복원 오차 기준)·거대 값·희소도·coverage·권고, HF·Lightning 콜백 | 확정 |
| [0038](0038-spill-engine-v01.md) | 2026-09-25 | design+milestone | A1b 저장 엔진: xxh3-128 다이제스트, 중간 버퍼 없는 병렬 읽기·쓰기, 0600 방출 파일, 저우선순위 쓰기 풀, codec pack/unpack, buffer protocol 바인딩, E008 프로토타입 제거 | 확정 |
| [0039](0039-hibernate-v01.md) | 2026-09-25 | design+milestone+experiment | N1c β hibernate와 노트북 통합, M1 시연: GPT-2 498MB를 SSD 쓰기 없이 해제·비트 동일 복원 3/3, 깨우기 0.42s 대 다시 불러오기 0.56s | 확정 (CUDA 미검증) |
| [0040](0040-v01-development-build.md) | 2026-09-25 | milestone | v0.1 개발판 완성: 테스트 Python 97·Rust 21, 패키징 확인(미배포), 완료 조건 대비표, 배포 전 검증 V1~V5 | 확정 (검증 대기) |
| [0041](0041-completeness-review-d1-d3.md) | 2026-09-25 | survey+milestone | 자기 비판 3차(완성도 평가, 탐색적 시험 6종): 결함 D1(공유 저장공간이 조용히 깨짐)·D2(역전파 도중 동면)·D3(meta 텐서 예외 누출) 발견·수정, 회귀 테스트 5개 | 확정 |
| [0042](0042-github-ci-first-run.md) | 2026-09-25 | milestone+experiment | GitHub 저장소(`imhyensuk/memopro`, 비공개)와 첫 CI: Linux·Python 3.11·cgroup 512MiB 한도 인식 통과, macOS 가상머신의 MPS 오판정 발견 → 실제 할당으로 판정 | 확정 |
| [0043](0043-ci-minimal-usage.md) | 2026-09-25 | decision | CI 사용량 최소화: PR은 Linux만, macOS는 main push·수동 실행 때 1개 작업, 문서만 바뀌면 생략, 중복 실행 취소, 빌드·pip 캐시 | 확정 |
| [0044](0044-alpha-prep.md) | 2026-09-25 | milestone | 알파 준비: 버전 0.1.0a1, 배포 워크플로(태그·버전 일치, 깨끗한 환경 wheel 확인, 신뢰 게시, 수동 실행은 Linux wheel만), Colab CUDA 확인 노트북 | 확정 (배포 대기) |
| [0045](0045-colab-cuda-v2.md) | 2026-09-25 | experiment+milestone | 관문 V2 Colab T4: doctor CUDA 정확, host·source 동면 GPU 512MB 회수·비트 동일, D2 확인 / F1 회수량 음수 숨김 → 부호 있는 보고, F2 census 분류율 82.9% → cuBLAS 작업 공간 분류(2차 확인 대기) | V2 통과, F2 확인(→0046) |
| [0046](0046-colab-cuda-run2.md) | 2026-09-25 | experiment | Colab T4 2차: cuBLAS 작업 공간 독립 측정 일치(18,087,936), CUDA 분류율 100%, host의 RAM 증가 −494.8MB 보고, source의 임시 사본 약 19MB 관찰 → V2 완료 | 확정 |
| [0047](0047-verification-record-and-assessment.md) | 2026-09-25 | survey+milestone | 검증 과정 종합(8개 층: Rust 21·Python 105 테스트, 독립 기준 7종, CI 10회, 탐색적·실기 검증으로 결함 D1~D3·F1·F2 발견)과 완성도 평가: v0.1 약 80%, 가장 큰 공백은 제품 가치 검증 | 확정 |
| [0048](0048-defect-sweep.md) | 2026-09-25 | survey+milestone | 결함 소탕: 탐색적 시험 40종(3차례)으로 결함 16건(조용한 오류 6) 발견·수정. 가드(저장·복사·이동·스텝에서 먼저 깨움), 롤백, 약한 레지스트리, 잠금, 이름 변환 대응 / 관찰: CPU 매핑 가중치 정렬로 BLAS 결과 1e-4 차이(memopro 무관) | 확정 |

## 연구 질문 (Research Questions) — 논문 A 기준 (0033)

**논문 A (주력): "학습 중 메모리는 얼마나 낭비되는가: 학습 상태의 종류별 정보량 측정"**

- **RQ-A1 (필요 비트)** 학습 상태(가중치·그래디언트·옵티마이저 모멘트·저장 활성값)는 종류별로 실제 몇 비트가 필요한가? 무손실 기준과 학습 영향 기준으로 각각 본다.
- **RQ-A2 (변화 요인)** 그 낭비는 모델 크기, 학습 단계, 정밀도(fp32·bf16), 옵티마이저, 과제(처음부터 학습·파인튜닝)에 따라 어떻게 달라지는가?
- **RQ-A3 (OS 기준선)** OS의 범용 메모리 압축은 이 낭비를 얼마나 회수하는가?
- **RQ-A4 (집중 위치)** 낭비와 민감도는 어디에 몰려 있는가? (이상치, 거대 활성값, LayerNorm 관련 텐서)
- **논의** 원리적으로 그럴듯한 기법(α)은 왜 실패했는가? (선행 이론의 예측과 실측, 0029 D25)

**논문 B (v0.1 이후, 도구 논문)**: β + census의 설계, 실제 사용자 데이터(E010), 기존 도구 비교, 외부 사용자 평가.

### 이전 연구 질문 (보존)

이전 버전(플래너 노선, RQ1–RQ3)은 0003/0004 참조. 아래 RQ1~RQ3(α)은 0018에서 기각되었다. 부정적 결과는 논문 A Discussion의 한 절로 다룬다(0030 C1). RQ4는 E010(0032 I1)과 논문 B에서 다룬다.

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
- 동면: 회수 메모리, 복원 지연(ms), 압축률, **SSD 쓰기 바이트**(0032 H5)
- 센서스(논문 A): 범주별 바이트·필요 비트·무손실 압축률·최악 민감도, OS 압축 기준선 대비 회수량
