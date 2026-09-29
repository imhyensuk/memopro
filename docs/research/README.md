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
| [0049](0049-colab-run3-and-torch-compile.md) | 2026-09-25 | milestone+experiment | PR #2 병합, Colab 3차 검사 사전 등록(가드 CUDA 14·GPT-2 4·compile 7, 로컬 CPU 25/25), wheel 로컬 교차 빌드. torch.compile 결함 발견·수정: 동면 중 호출 시 dynamo 오류, 분할 뒤 조건 없는 eager 캐시로 이후 호출이 모두 eager(조용함) → 컴파일된 프레임 전에 깨움 | 확정 |
| [0050](0050-colab-cuda-run3.md) | 2026-09-26 | experiment | Colab T4 3차: 사전 등록 검사 25/25 통과(가드 CUDA 14·GPT-2 4·compile 7, CUDA graphs 포함, 비트 동일·재컴파일 0), 교차 빌드 wheel 동작 확인 / 관찰: census 적용률 98.9%(환경 변화, 원인 미식별), host의 RSS 증가량은 할당자 재사용에 따라 다름 | 확정 |
| [0051](0051-resurvey-gamma-run.md) | 2026-09-26 | survey | v0.3 착수 조건(0030 C5): γ·`memopro run` 중복성 재조사. OS 압박 신호로 PyTorch 구성을 되돌릴 수 있게 바꾸는 런타임은 찾지 못함(Tri-Accel, eLLM, LOCAL, OOM 재시도와 차이), `run`은 측정 래퍼·import 패치와 부분 중복(신규성 주장 없음). 연동 도구 로컬 확인(torchao int4 불가, bitsandbytes MPS 가능, MemTracker 할당 없는 추정) / 관찰: 이 Mac은 압박 "경고" 상태 | 확정 |
| [0052](0052-build-v02-v03-decision.md) | 2026-09-26 | decision | 사용자 지시로 v0.2·v0.3을 Gγ 전에 제작. A2 완료 조건 사전 등록(로컬 소형 + Colab 7B·±15%·실제 OOM), 품질 4등급·선호, `train_session`의 OOM 재시도와 옵티마이저 교체는 제안만, census 정밀 모드 정의, γ는 안전 지점에서만, `run`은 명시 선택을 바꾸지 않음, Windows CI | 확정 |
| [0053](0053-build-v02-v03.md) | 2026-09-26 | milestone+experiment | v0.2·v0.3 제작: `load`·`check`·`optimize`·`train_session`·census 정밀 모드·pressure·γ·`run`. A2-2 기준 수정(AdamW가 fp32 합산 차이를 3e-4로 증폭 → 그래디언트·SGD 기준). 테스트 166, 탐색 12/12, 탐색 중 고친 것(fp16 손실 스케일링 등). Colab 4차 사전 등록 / 관찰: 경고 상시로 γ 예산 계수 보정 필요, 정밀·경량 필요 비트 차이 | 확정 |
| [0054](0054-colab-run4-access.md) | 2026-09-26 | experiment | Colab T4 4차: 7B(Qwen2.5-7B) int8 로드·제한된 장치 학습(SGD 1.6e-7)·load 4단계·활성값 오프로드 통과, **`check` ±15%는 불합격(4/6, 작은 배치 학습 과소 예측)** — accelerate보다는 6/6 정확(평균 오차 11.8% 대 37.7%). 원인 하나(가짜 텐서는 foreach 미선택) 수정, 5차 사전 등록. 결함: 실패한 로드가 메모리를 쥔 채 다음 구성 시도 → 수정 | 확정 |
| [0055](0055-colab-run5-check-diagnosis.md) | 2026-09-26 | experiment | Colab T4 5차: `check` 재검증 불합격(3/6) + **측정 오염 발견**(앞 경우 메모리가 남아 누적, MemTracker 순환 참조) → 경우별 프로세스 격리. 원인 셋 확인·수정: 첫 스텝 추적(→ 둘째 스텝), 출력 보유로 logits 생존, foreach 미선택. CPU에서 실제 추적과 +0.00% 일치, 6차 사전 등록 | 확정 |
| [0056](0056-colab-run6-check-pass.md) | 2026-09-26 | experiment | Colab T4 6차: **`check` ±15% 통과(6/6)** — 학습 −0.2~−1.2%, 추론 +5%, 예측이 CUDA 실제 텐서 추적과 0.003% 이내 일치. 세 번 만의 통과에 대한 평가(보정 계수 없음, 독립 기준). **A2 완료 조건 모두 충족**(원래 M1 3B 조건만 열림) | 확정 |
| [0057](0057-v01-v03-consolidated-record.md) | 2026-09-26 | survey+milestone | v0.1~v0.3 종합 기록(PR #4 병합 시점): 현재 기능 전체와 근거, 테스트 Python 173·Rust 24, CI 3개 OS, 탐색 63종, T4 6회, 결함 목록, 사전 등록 판정(실패 포함), 미확인 영역. 완성도: 설계 범위 약 90%, v1.0 약 45%, 제품 가치 검증 10% 미만 | 확정 |
| [0058](0058-e011-preregistration.md) | 2026-09-26 | experiment (사전 등록) | E011 β 대 macOS 스왑·압축기. 파일럿에서 발견: 조사가 가드를 건드려 모델을 깨움(수정), CPU fp32 가중치는 mmap, macOS libmalloc이 해제한 메모리를 캐시에 붙잡음(`MallocLargeCache=0`이면 반환). 조건 cpu/mmap·cpu/anon·mps, 방법 os·source·compress·reload(+진단 source_nocache), 기준 D0~D5 | 확정 |
| [0059](0059-budget-forms-extension.md) | 2026-09-26 | decision+design | 예산 형식 확장: 남기기 `-2GB`, 범위 `2GB..6GB`(최소 미달 시 멈춤), 강제 `6GB!`, 풀별 모든 형식과 디스크 풀, `{use,min,max}`, `budget_basis`(conservative·os·total), 크기 여유분, `memopro.using()`. 측정값을 넘는 것은 `!`와 basis뿐(0035·0052 E3을 명시적 선택에 한해 완화) | 확정 |
| [0060](0060-e011-results.md) | 2026-09-26 | experiment | E011 결과: D0 비트 동일 73/73 ✅, D3 복귀 지연 OS 대비 0.33×(CPU)·0.45×(MPS) ✅, D2 다른 작업 반응성 이득 없음 ❌, D1 즉시 반환 실패(−0.34·−0.62) ❌ → 결함 F-E011-1(libmalloc 캐시·대조 버퍼), 해결 선택지 F1~F5 사용자 결정 대기. 페이지 LZ4는 파라미터에 1.00, memopro 0.84. 잠자기 중단 8개 시행은 사전 규칙으로 재실행 | 확정 |
| [0061](0061-f1-f4-malloc-cache-fix.md) | 2026-09-27 | decision+milestone | F-E011-1 대응: F1 `source` 원본 대조를 32MiB 조각 단위로(청크 해시 결합, Rust `digest_parts/combine`), F4 macOS `MallocLargeCache` 안내(doctor, 보고서 1회)와 `memopro run` 재시작(`--keep-malloc-cache`). 테스트 10개 | 확정 |
| [0062](0062-e014-preregistration.md) | 2026-09-27 | experiment (사전 등록) | E014 "적은 메모리에서 쓸 만한가": 추론(1.5B·3B × naive·memopro 4방법), 학습, 모델 번갈아 쓰기, F1·F4 확인. 쾌적 = 5 tok/s·불러오기 120s·스왑 1GiB·탐침 p95 ≤ 유휴 2배. 파일럿: MPS는 그래픽 드라이버가 해제된 메모리를 붙잡음, 기본 설정은 1.5B를 거부 | 확정 |
| [0063](0063-e014-results.md) | 2026-09-27 | experiment | E014 결과: 학습 일반 루프 OOM 3/3 → `train_session` 완주 3/3 ✅. 번갈아 쓰기 둘 다 올려 두면 중단, β는 다시 불러오기의 0.90배(C5 ❌). 쾌적한 추론 없음 ❌. **기본 설정은 1.5B·3B 모두 `BudgetExceeded`** → 기본값 결정 대기(D-a~D-d). F4로 CPU 반환 0.965 ✅, 대가 1.05 ✅. 같은 bf16에서 memopro 불러오기는 스왑·탐침 1/3 | 확정 |
| [0064](0064-defaults-da-dd.md) | 2026-09-28 | decision+milestone | 기본값 문제 대응(사용자 결정): D-a(필수) 들어갈 구성이 없으면 계획을 다시 세워 **실제로 불러와지는 설정만** 안내(스왑 예상량, 빈 메모리 우선 → 스왑 적은 순), `BudgetExceeded.suggestions`, `check`·`optimize`·`run`에도. D-d(선택) `fallback="stored"`: 경고하고 원래 형식으로 장치에 바로 불러옴(`optimize`는 그대로 둠). 제안 설정이 `load` 인자로 안 먹던 결함 수정. 테스트 12개 | 확정 |
| [0065](0065-status-after-e011-e014.md) | 2026-09-28 | survey+milestone | 현재 상태 종합(main `b2a182d`): 완성도 설계 범위 약 92%·v1.0 약 48%·제품 가치 검증 약 15%, CUDA·M1 실측 표, 기능 목록, "가능 대 쾌적" 평가 | 확정 |
| [0066](0066-comfort-methodology-rcr.md) | 2026-09-28 | survey+design | 8GB 쾌적화 방법론: 실측(메모리 60.3GB/s, SSD 2.45GB/s, 흩어진 page-in ≈0.13GB/s)과 디코딩 모델, 진단 D1~D6, 원칙 R1~R5, 1층(기존 기법 연결), 2층 **RCR**(텐서를 재구성 비용 등급 F·P·C·A로 나눠 OS 메모리 종류에 사상 + 쾌적 제어기, 조합은 조사 범위 내 미발견), 비트플레인 자기 추측(중복 중간~높음), 학습 경로, 검증 계획 E015~E018(관문 G-P·G-F) | 초안(사용자 결정 대기) |
| [0067](0067-e015-preregistration.md) | 2026-09-28 | experiment (사전 등록) | E015 RCR 기초 측정: Q1 재읽기 비용(5방법), Q2 purgeable(G-P), Q3 무복사 MPS 텐서(G-F, Objective-C 도우미 + DLPack), Q4 MPS 저비트 커널. 파일럿: 무복사 텐서 비트 동일, bnb nf4는 bf16보다 느림 | 확정 |
| [0068](0068-e015-results.md) | 2026-09-28 | experiment | E015 결과: **G-F ✅**(파일 매핑 MPS 텐서 비트 동일·int4 비트 동일, footprint 증가 없음, 압박 시 OS가 48~68% 회수, 재계산 비트 동일), **G-P ✅**(purgeable이 익명 대부분이 밀리기 1~2초 전 쓰기 없이 비워짐), Q1 폴트 재읽기 0.28GB/s·데우기 0.74GB/s·원시 2.49GB/s(선읽기 필요), **Q4 torch int4pack이 bf16의 1.63배, bnb nf4는 0.35배** → int4 백엔드 교체 결정 대기, RS6 해제 불필요 | 확정 |
| [0069](0069-int4-backend-torch-int4pack.md) | 2026-09-28 | decision+milestone | MPS int4 백엔드를 torch `_weight_int4pack_mm`(+torchao group-wise 양자화, 그룹 64)로 교체: CPU에 mmap으로 불러와 층별 변환, 헤드·임베딩은 bf16. 1.5B int4 **4.9 → 14.7 tok/s**(bf16 11.0), 장치 가중치 1.16GiB. MPS 드라이버의 0.67GB 잔여는 남음 | 확정 |
| [0070](0070-e016-preregistration.md) | 2026-09-28 | experiment (사전 등록) | E016 RCR 추론 시제품: F 등급 bf16(원본 safetensors를 그대로 무복사 매핑)·int4(한 번 변환한 정렬 F 파일)·선읽기, 방법 5개 × 1.5B·3B, A(평소)·B(4GiB 압박) 단계, 기준 E1·C2·H1~H3. 탐침 기준 max(유휴×2, 24.1ms)는 파일럿 후 결정 | 확정 |
| [0071](0071-e016-results.md) | 2026-09-28 | experiment | E016 결과: **rcr_int4 1.5B 18.0·3B 10.6 tok/s, footprint 0.53GB, 스왑 0, 불러오기 0.4s, C2 ✅(두 모델)**, 압박에서 속도 83%·58% 유지(memopro_int4 41%·39%). memopro_int4(0069)도 C2 ✅. rcr_bf16은 1.5B ✅, 3B는 0.24 tok/s로 붕괴(여유 초과). 관찰: 압박 중 RCR은 이웃 프로세스를 스왑으로 밀어냄 → 제어기(E017) 필요. F 등급 기능화 결정 대기 | 확정 |
| [0072](0072-fclass-feature.md) | 2026-09-28 | decision+milestone | RCR F 등급 기능화: `residency="file"`(구성 선택은 그대로, 가중치를 깨끗한 파일 페이지로). stored는 원본 safetensors 매핑(정렬 안 되면 정렬 캐시), half·int4는 F 캐시(만들 때 `disk_writes="allow"`·디스크 예산, 0600, 원본 변경 시 무효). Rust `FileMap`(mmap·mincore·pread·Metal 무복사, 새 크레이트 없음). 1.5B bf16 매핑 0.73s, int4 캐시 재사용 0.13s. 테스트 Rust 3·Python 7 | 확정 |
| [0073](0073-e017-preregistration.md) | 2026-09-28 | experiment (사전 등록) | E017 쾌적 제어기: bf16·int4 F 매핑을 함께 쥐고 공유 KV로 토큰 사이 전환, 신호 = 시스템 스왑 속도(이웃 피해)·자기 재읽기(붕괴), 조치 = 정밀도 내리기·속도 조절·복귀. 파일럿: OS는 활발한 우리 파일 페이지를 지키고 이웃을 스왑으로 밀어냄, 압박 비용은 대부분 재정착 구간. 기준 K1~K4 | 확정 |
| [0074](0074-e017-results.md) | 2026-09-28 | experiment | E017 결과(사용자 요청 중단·재개, 3B bf16 기준점 1회): **K2 ✅** 3B bf16 0.24 → 10.5 tok/s(40배), 1.5B 압박 중 6.6 → 11.7. 1.5B bf16 복귀 2/3. **결함: 예열 때 첫 읽기를 붕괴로 오판해 A 전에 int4로 내려감** → K3·K4 일부 실패, 기능화 보류, 수정 후 E017b 제안. 정상 상태 이웃 피해는 작고 비일관적. `load(residency="file")` footprint 1.5GB 조사 필요 | 확정 |
| [0075](0075-e017b-preregistration.md) | 2026-09-28 | experiment (사전 등록) | E017b: 제어기 수정(시작 정밀도를 여유로 정함, 예열 동안 판단 안 함, 전환 직후 한 창은 page-in 무시) 후 1.5B bf16 대 bf16_ctl, 3B bf16_ctl 재측정. 기준 K0(수정 작동)~K4, S(3B는 int4로 시작) | 확정 |
| [0076](0076-e017b-results.md) | 2026-09-28 | experiment | E017b 결과: 첫 읽기 오판은 고쳐짐(3B는 int4로 시작, S ✅), 압박 중 1.5B 6.9 → 15.2 tok/s, 재정착 탐침 1.9초 → 0.14초. **K0 ❌**: 쉴 때도 0.16초 창의 시스템 스왑 속도 신호가 켜져 2/3에서 bf16을 떠남(제어기 없는 bf16도 쉴 때 스왑 0.33~0.82GB 증가), 시작·복귀가 낙관적인 `kernel_available`을 씀. 수정안 Q1~Q3 사용자 결정 대기 | 확정 |
| [0077](0077-e017c-preregistration.md) | 2026-09-28 | experiment (사전 등록) + decision | 사용자 결정: Q1(1초 이상 구간 스왑 증가 ≥ 64MB) + Q2(보수적 가용량) 적용, **쾌적 우선 모드 기본**(품질 우선은 선택). E017c: 1.5B bf16·int4·제어기, 3B int4·제어기. 기준 C0(쉴 때 안정)·C1(E014 쾌적)·K2·K4·S | 확정 |
| [0078](0078-e017c-results.md) | 2026-09-28 | experiment | E017c 결과: 쾌적 우선 모드가 쉴 때 안정(C0 ✅), E014 쾌적 기준 충족(C1 ✅, 스왑 0), 압박 중 1.5B 11.8 tok/s(bf16 6.4), 이웃 반응성 p95 474ms(bf16 2.4초). **K4 ❌**: 시작 정밀도가 불러올 때의 여유에 따라 달라져(1.5B int4 ×2, bf16 ×1) 출력이 바뀜 → 기능화 보류, 선택지 P1~P3. 3B 제어기 방법 footprint 0.49GB 대 int4 단독 1.47GB(재정착 비용도 작음) → footprint 조사 우선 | 확정 |
| [0079](0079-controller-p1-p3.md) | 2026-09-28 | decision + implementation | 사용자 결정 P1+P3: 쾌적 우선은 항상 int4로 시작하고 올리지 않음(출력 결정적), 여유에 따른 bf16은 품질 우선에서만(경고), 고른 정밀도와 이유를 항상 보고, `precision=` 고정. 시제품에만 적용, 기능화는 짧은 재측정 뒤 | 확정 |
| [0080](0080-e019-footprint.md) | 2026-09-28 | experiment | E019 footprint 조사: E017 계열의 추가 1GB는 **PyTorch MPS 할당기가 10MiB 이상 할당 때 잡는 1GiB 힙**(`kXLargeHeap`, 압박 아닐 때만). F 등급 탓 아님(`lm_head` 복제해도 같음). bf16 매핑이 드라이버 할당을 저수위선 위로 올려 힙이 안 생겼던 것. `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1`이면 1.5B·3B 모두 1.5 → 0.49GB, 속도·출력 동일(0.3은 1.5B int4에 효과 없음, 0.01은 약간 느림). 반영 선택지 W1~W4 사용자 결정 대기 | 확정 |
| [0081](0081-mps-heap-w1-w2.md) | 2026-09-28 | implementation | 사용자 결정 W1+W2: `memopro run`이 Apple silicon에서 `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.1`로 다시 실행(`MallocLargeCache=0`과 한 번에, 사용자 값 유지, `--keep-mps-heap`), doctor·파일 기반 load 안내(한 번). 비율은 MPS 사용량이 권장 최대치의 10%를 넘어야 효과. 끝에서 끝까지 1.5B int4 1,470 → 461MiB. Python 238·Rust 28 통과 | 확정 |
| [0082](0082-controller-not-a-feature.md) | 2026-09-28 | decision | 쾌적 제어기는 라이브러리 기능으로 만들지 않고 연구 결과로 남김(쾌적 우선에서는 거의 조치 없음, 쾌적함은 정적 선택·파일 상주·할당기 설정에서 나옴, 동적 전환은 재현성 비용). W3 취소 | 확정 |
| [0083](0083-e020-preregistration.md) | 2026-09-28 | experiment (사전 등록) | E020 int4 품질: WikiText-2 앞 32×2048토큰, bf16 대비 PPL·KL·1순위 일치, 그룹 32/64/128, 긴 입력 속도. 파일럿: 파일 캐시 = 즉석 g64(완전히 같음), 풀어서 계산 ≈ 커널(PPL 0.06% 차이), **int4 커널은 2048토큰 입력에서 bf16보다 5.6배 느림**. 기준 G1(등급)·G2(그룹)·R2(혼합 방식) | 확정 |
| [0084](0084-e020-results.md) | 2026-09-28 | experiment | E020 결과: int4 g64 PPL 증가 **1.5B +9.2%, 3B +21.1%**(1순위 일치 85%·81%), g32는 +5.7%·+7.6%(가중치 +7~8%), g128 +12.4%·+15.4%. int4 커널은 길이에 정비례(1024토큰에서 bf16의 6.2배·3.7배 느림, 2048토큰 첫 토큰 38초·71초). G1(경고)·G2(g32)·R2(혼합 경로) 발동 → 제안 Q-a~Q-c 사용자 결정 대기 | 확정 |
| [0085](0085-int4-qa-qb-qc.md) | 2026-09-29 | implementation | 사용자 결정 Q-a~Q-c: int4 불러오기 보고에 품질 비용(PPL +5.7~7.6%), MPS int4 기본 그룹 32(디코딩 −2~4%, 크기 추정 반영), 입력 160행 이상은 int4를 풀어 bf16 행렬곱(묶음 형식 = [N,K] 코드 int32당 8개). 1.5B 1024토큰 프롬프트 29.2 → 8.7초(같은 토큰). 3B 첫 속도 측정은 동시 테스트로 오염돼 다시 잼. Python 246 통과 | 확정 |
| [0086](0086-e010-preregistration.md) | 2026-09-29 | experiment (사전 등록) + tool | E010: 참여자용 단일 파일 수집 도구(E006 유휴 판정 + 이름 해시, 원본 파일 여부, OS 메모리·스왑, GPU, OOM 표시, 코드·값·이름 미기록) 검증 13/13 통과, 참여자 안내(한·영), 지표 S2~S4와 Gβ 판정 규칙(참여자 3명·세션 8개 이상). **수집은 사용자·동료가 해야 함.** 옛 int4 g64 캐시 3.4GB 삭제 | 확정 |
| [0087](0087-e013a-preregistration.md) | 2026-09-29 | experiment (사전 등록) | E013a: γ의 macOS 압박 수준 신호 보정. 1초마다 수준·스왑·가용량·반응성 탐침, 평온과 압박 1~4GiB ×2회. 기준 사실 = 탐침 > 24.1ms. 후보 L(현재 신호)·S(스왑 64MB/1초)·A·K·S∨L, 쓸 만함 = 민감도 ≥ 0.8·특이도 ≥ 0.9 | 확정 |
| [0088](0088-e013a-results.md) | 2026-09-29 | experiment | E013a 결과: macOS 압박 수준은 압박 중 87% 켜지고 평온·해제 뒤 0%(0053의 "평온해도 경고"는 오늘 재현 안 됨), 그러나 끊김(탐침 > 24.1ms)은 870초 중 5초뿐이라 특이도 0.54. 스왑 신호 민감도 0.40. **쓸 만한 신호 없음(규칙 3)** → G1(γ 실험 기능, macOS `run` 기본 끔)·G2(경고엔 손실 없는 조치만) 사용자 결정 대기. SSD 쓰기 15분 9.2GB | 확정 |
| [0089](0089-gamma-g1.md) | 2026-09-29 | implementation | 사용자 결정 G1: γ는 실험 기능, macOS의 `memopro run`은 γ를 기본으로 켜지 않음(`--elastic`으로 켬, 기본으로 꺼지면 이유를 보고). `enable()` 직접 사용과 γ 동작은 그대로. 시험 5개 추가 | 확정 |
| [0090](0090-status-evaluation.md) | 2026-09-29 | assessment | 완성도 평가: 기능 약 90%, 기술 검증 약 75%, 실사용 가치 입증 약 35%, 사용성 약 60%, 출시 준비 약 45%(연구자 판단). 강점(숫자로 입증된 개선, 정직한 기본값), 약점(E010 없음, 7B·학습 쾌적함 미입증, MPS 편중, 마지막 CUDA 검증이 0056), 출시까지의 순서 | 확정 |
| [0091](0091-colab-run7-preregistration.md) | 2026-09-29 | experiment (사전 등록) + tool | E021(Colab 7차) 준비: 독립 셀 3개 노트북(`examples/colab_t4_heavy.ipynb`) — 학습(GPT-2·0.5B·1.5B 전체 미세조정, 일반·AMP·ckpt·accelerate 대 memopro, 작은 GPU 흉내, check, 7B QLoRA), 로컬 AI(1.5B~14B, HF fp16·bnb8·bnb4 대 memopro), 여러 모델(동면 대 다시 불러오기, `memopro run`, 회귀 7종). Drive 캐시·결과·이어하기, 경우마다 새 프로세스. 로컬 MPS로 끝까지 확인(MPS `.to("cpu", float64)`가 틀린 값을 내는 문제 발견). 기준 R1·T1~T4·I1·I2·M1~M3 | 확정 |
| [0092](0092-colab-notebooks-split.md) | 2026-09-29 | tool | Colab 7차 노트북을 파트별 3개로 분리: `colab_t4_train.ipynb`(학습·개발), `colab_t4_infer.ipynb`(로컬 AI 모델), `colab_t4_multi.ipynb`(여러 모델·프로세스). 각 파일은 안내 + 통합 셀 1개, 코드는 이전과 같음(사전 등록 0091 그대로) | 확정 |
| [0093](0093-colab-run7-results.md) | 2026-09-29 | experiment | E021(Colab 7차) 결과: R1 회귀 7/7 ✅, `train_session`은 정확한 전체 배치와 1e-6~1e-8로 같고 plain·AMP·ckpt가 OOM인 곳(0.5B 배치 8, GPT-2 30%)에서 배치 그대로 완주, `check` 5/5 "맞지 않음" 정확, 안 되는 학습은 제안과 함께 거절, 추론은 OOM 없이 14B까지(int4 9.0 tok/s). T1은 plain이 전부 OOM이라 판정 불가. **결함**: D1 동결 파라미터 계획 실패, D2 T4에서 bf16 선택(긴 프롬프트 3.7~5.4배 느림), D3 bnb int8 속도 순위, D4 Linux γ가 불러오기 I/O를 압박으로 오판해 `run` 정책 무력화, 동면은 다시 불러오기의 2.1배 느림 | 확정 |
| [0094](0094-e021-fixes-d1-d4.md) | 2026-09-29 | implementation | 사용자 결정 D1~D4: 동결 파라미터에서 CUDA 측정으로 계획(기울기·Adam 상태는 학습 파라미터만), sm < 8 CUDA에서 bf16 대신 fp16 우선(lossless는 bf16, `run`은 원본이 들어가면 개입 안 함), bnb int8 속도 등급 2, `memopro run`의 γ 모든 플랫폼 기본 끔. 노트북 회귀 기대값·다시 불러오기 참조 수정. 시험 261개 통과, CUDA 재측정 전 | 확정 |
| [0095](0095-objective-evaluation.md) | 2026-09-29 | assessment | 모든 실험 근거의 객관 평가: 추론 부분 입증(장치 약 2배까지 자동·OOM 없음, 60%), 학습 부분 입증(활성값 초과는 정확히 해결, 파라미터 초과는 제안만, 55%), 개발 편의 입증(70%), 연구 코어 약함(α 반증, β 정확하나 가치 미입증, γ 부정적, 25%), 제품 준비 45%, 가중 약 53%. 기존 도구 대비 새 능력은 없고 가치는 자동화·안전·정확성. **정정**: 0063에서 β는 다시 불러오기의 0.90배 시간(10% 빠름)이었음(0090의 "느렸다"는 틀림) | 확정 |
| [0096](0096-e022-remeasure-preregistration.md) | 2026-09-29 | experiment (사전 등록) + tool | E022: 0094 수정 확인용 통합 셀 노트북(`colab_t4_remeasure.ipynb`) — 회귀 7종, 1.5B·3B fp16 선택과 첫 토큰 시간(D2), 7B low의 int4와 기본 int8(D3), 7B QLoRA 계획(D1), `memopro run` 7B 정책 적용·1.5B 비개입(D4). 요약에 판정 표 자동 계산, 진행 로그 저장. 새 sdist 필요 | 확정 |
| [0097](0097-decision-open-source.md) | 2026-09-29 | decision | 사용자 결정: 무료 오픈소스로 배포(`MIT OR Apache-2.0` 유지). SaaS·비공개 유료는 택하지 않음(사용자 기기에서 도는 목적과 충돌, 무료 대체재, 가치 미입증). 오픈 코어는 사용자가 생긴 뒤 새 결정으로만. 게시·공개 시점은 여전히 사용자 확인, CLA·기관 규정 확인 권고 | 확정 |
| [0098](0098-e022-remeasure-results.md) | 2026-09-29 | experiment | E022 결과: 판정 8/8 pass로 D1~D4 닫음 — T4 fp16 선택으로 2048토큰 첫 토큰 HF fp16의 1.04~1.07배(7차 대비 3.6~5.0배 빨라짐, PPL 동일), 7B low는 int4로 15.8 tok/s(HF bnb4 14.0), QLoRA 계획 완주(손실 차이 ≤1.2e-3, 메모리 −23%, 속도 0.88배), `run` 7B 정책 적용·1.5B 비개입. **새 결함** D8 학습 계획이 이미 올라간 가중치를 예산에서 다시 빼 micro-batch를 과소 선택(D5와 함께 수정 필요), D9 CUDA bnb에 MPS 품질 수치 문구. 관찰: 기본 7B int8은 int4의 1/2.9 속도 | 확정 |
| [0099](0099-e022-fixes-d8-d5-d9-o1.md) | 2026-09-29 | implementation | 사용자 결정 D8+D5, D9, O1: 이미 메모리에 있는 모델·옵티마이저 상태를 측정값에 다시 더한 뒤 예산 설정 적용(크기 상한은 모델 포함 총량, `train_session`·`check`·`optimize`), CUDA 가용량에 torch 캐시 여유 포함, 계획에 할당기 여유 1.25배와 고른 분할(재시도 포함), int4 품질 문구 백엔드별(bnb +7.6~8.4%, int4pack +5.7~7.6%), 기본 품질이 bnb int8을 고르면 `quality='low'`의 int4 속도 안내(선택은 불변). 시험 270개 통과, CUDA 재측정 전 | 확정 |
| [0100](0100-code-review-after-0099.md) | 2026-09-29 | implementation | 재측정 전 정밀 검토: 0099 예산 규칙의 빈틈 보완 — `total` 기준·통합 메모리에서 보유 메모리 이중 계산(B1·B2), γ 축소는 보유분 밖만(B3), bnb 양자화 보조 텐서 포함·같은 메모리 한 번(B4), 첫 GPU만(B5), 장치 예산 0을 호스트로 대체하던 `or`(B6), 옵티마이저별 앞으로 생길 상태(B7), `run` 안내는 `--quality low`. 노트북: 같은 모델 한 번만 복사, 끊긴 복사 방지, 빌드 확인 후 중단. 시험 277개 통과 | 확정 |
| [0101](0101-e023-remeasure2-preregistration.md) | 2026-09-29 | experiment (사전 등록) + tool | E023: 통합 셀 노트북 `colab_t4_remeasure2.ipynb` — 회귀+D9, GPT-2 고른 분할·정확(T1), 30% 상한 OOM 재시도 없음(T2), 0.5B micro ≥ 2·정확(T3), 7B QLoRA micro ≥ 2·표준의 0.95배 이상(T4), 7B 기본 int8 유지와 int4 안내(O1). 약 45~60분, 새 sdist 필요 | 확정 |
| [0102](0102-e023-remeasure2-results.md) | 2026-09-29 | experiment | E023 결과: 판정 8/8 pass, 예측한 계획 크기와 모두 일치 — GPT-2 8+8(HF ckpt 대비 손실 1.7e-7, 1.15배 빠름), 30% 상한 OOM 재시도 없이 2, 0.5B micro 2로 1.09배, 7B QLoRA micro 4로 표준 방식과 같은 227 tok/s(E022 대비 1.17배), int4 안내·bnb 품질 문구 확인, 보유 메모리 7.63GiB = torch 할당 7.64GiB. **관찰**: 예약 기준 활성값 배수 1.27~1.29로 여유 계수 1.25가 빠듯(F1, 제안 1.3), 시작 때 캐시 1.99GiB 중 1.67GiB가 재사용되지 않아 QLoRA 최대 예약이 예산을 0.15GiB 넘음(F2, 제안: 시작 때 캐시 비우고 실측) | 확정 |
| [0103](0103-pa-pb-margin-and-cache.md) | 2026-09-29 | implementation | 사용자 결정 P-a·P-b: 여유 계수 1.25 → 1.3(E023 예약 배수 1.27~1.29), `train_session`은 CUDA 모델이면 예산 측정 전에 캐시를 비우고(`release_cuda_cache`, 보고에 돌려준 양) 가용량에는 통째로 빈 세그먼트만 셈(`releasable_cache` = reserved − active − inactive_split, 조각은 제외). 워커는 세션 진입 뒤 상태(`session_measured`) 기록. 시험 280개 통과, CUDA 재측정 전 | 확정 |
| [0104](0104-e024-qlora-preregistration.md) | 2026-09-29 | experiment (사전 등록) + tool | E024: 7B QLoRA만 재는 통합 셀 노트북 `colab_t4_qlora.ipynb` — Q1 OOM 재시도 없이 완주, Q2 최대 예약 ≤ memopro 예산(E023 13.90 > 13.75GiB), Q3 표준 방식의 0.9배 이상·손실 ≤ 5e-3. 캐시 반환 전후 상태를 요약에 싣는다. 약 20~25분, 새 sdist 필요 | 확정 |
| [0105](0105-e024-qlora-results.md) | 2026-09-29 | experiment | E024 결과: 판정 3/3 pass — 7B QLoRA가 micro 2(2+2)로 OOM 재시도 없이, 최대 예약 11.00GiB ≤ 예산 11.96GiB(E023은 초과), 표준 방식의 0.97배(229 대 235 tok/s), 손실 1.2e-3, GPU 사용은 표준보다 2.2GiB 적음. 시작 때 캐시 1.99GiB 중 비울 수 있던 것은 4MiB — 거의 전부 조각(E023 F2 확인, 0103의 측정 방식이 맞음) | 확정 |
| [0106](0106-objective-evaluation-2.md) | 2026-09-29 | assessment | 객관 평가 2(E022~E024 반영): 추론 65%(T4 fp16·int4 결함 해소, 2배 초과 미입증), 학습 62%(QLoRA 0.97배·예산 준수, 파라미터 초과 학습은 범위 밖), 개발 편의 72%, 연구 코어 25%(변화 없음), 제품 준비 50%, 가중 약 58%(0095의 53%). 오른 이유는 결함 수정의 실측 확인. 다음: 2배 초과 시험, E010, D6, 배포 준비 | 확정 |
| [0107](0107-direction-goal-redefinition.md) | 2026-09-29 | decision | 사용자 승인 목표 재정의: 궁극 = 32GB가 필요하던 작업(AI·비AI)을 8~16GB에서 구동. G2′ 작업 집합이 메모리 안이면 필요 메모리의 1/2~1/4에서, 무손실 기본·상한 보장·느려짐 예측. G3 주 목표 승격: 버퍼마다 두기·압축·원본 재읽기·재계산·이동을 고르는 새 Rust 런타임. 기존 성과(접근 계층, β, 방출 엔진, F 등급, census, 실험) 계승. 0010·0012·0033 일부 대체 | 확정 |
| [0108](0108-survey-new-runtime-prior-art.md) | 2026-09-29 | survey | 새 런타임(C-R) 선행 연구 13개 주제: Williams 2025·Cook-Mertz(이론), Capuchin·POET(스왑/페이징+재계산, 가장 가까움), DTR·Checkmate·RevNet, AIFM·Mira·Atlas(원격 메모리), UMap·ExtMEM·Lightswap(사용자 공간 페이징), TMO·SDFM(커널 압축·오프로드), MEMPHIS(계보+메모리 관리, SystemDS), Dask, CLA, Mesh, LLM in a flash·mzCache. 판정: 개별 기법은 모두 있음, 다섯 선택지 통합·분야 무관·개인 기기 계층·상한 보장+느려짐 예측의 결합은 조사 범위에서 발견되지 않음(조합의 새로움). 재계산은 계보가 알려진 버퍼만 | 확정 |
| [0109](0109-runtime-design-draft.md) | 2026-09-29 | design (초안) | 런타임 C-R 설계 초안 v0.1([runtime.md](../design/runtime.md)): 다섯 무손실 행동(두기·압축·원본 재읽기·재계산·방출)과 상태 기계, recipe(Source·Lineage·Seed), 입장 제어로 상한 보장, 실측 비용 점수, 전용 아레나, 소유권 기반 스레드 안전, 원칙 R1~R8·불변식 I1~I7, Rust·C ABI·Python(`memopro.rt`) 인터페이스, 기존 spill·codec·ledger·residency·β 재사용, 3단계 관문(E025 G-R1 비AI 1/2, G-R2 AI 1/4, G-R3 Linux 투명). 결정 대기: RS7 해제, 대표 비AI 작업, 이름 | 초안 |
| [0110](0110-decision-no-ssd-writes-rs7-autonomy.md) | 2026-09-29 | decision | 사용자 승인: 런타임은 SSD에 쓰지 않음(방출·파일 캐시 제외, 원본 재읽기는 페이지 캐시 우회), 실험도 스왑·새 파일 없이 실제 RAM 안에서(스왑 전후 기록), RS7 해제(안전 규칙 R4·I2·I6), 1단계 작업 = 큰 숫자 배열 파이프라인, 이름 `memopro.rt`/`memopro::rt`/`mp_`, 끝까지 자율 진행(관문 실패 시 멈추고 보고, 테스트 통과 PR은 직접 병합), 다운로드 없음(로컬 모델만) | 확정 |
| [0111](0111-e025-preregistration.md) | 2026-09-29 | experiment (사전 등록) | E025 = 관문 G-R1: 로컬 safetensors 데이터 영역(GPT-2 523MiB f32, Qwen 1.5B 2.88GiB·3B 5.75GiB bf16)을 16MiB 블록의 큰 배열로 보고 3회 통계. naive·stream·stream_nc·memmap·memopro(W/2·W/4, 1GiB·512MiB), 파생 float32 512MiB 압축 경우. H1 상한, H2 비트 동일, H3 쓰기 없음, H4 못 돌던 것, H5 수동 스트리밍의 1.1배 이하, H7 순환 훑기 재읽기, H8 압축. 실행 전 수정 2건(H7 블록 보정, 큰 데이터에서 페이지 캐시 방법 제외) | 확정 |
| [0112](0112-runtime-phase1-build.md) | 2026-09-29 | implementation | 런타임 C-R 1단계: Rust `memopro::rt`(쪽 단위 mmap `Region`, 캐시 우회 `SourceFile`, 입장 제어로 상한 보장, 버림(원본)·압축(조각 단위로 쪽 반환) 두 행동, 실측 비용 × 재사용 주기 점수로 순환 훑기 MRU, 배타 `pin`·공유 `pin_shared`, `Error::Budget`), Python `memopro.rt`(`Runtime`, `add_file`, `load_npy`, `alloc`, 복사 없는 NumPy `view`·`apply`, bfloat16). Rust 16개·Python 10개 시험 추가, 전체 통과 | 확정 |
| [0113](0113-e025-results.md) | 2026-09-30 | experiment | E025 결과: 관문 G-R1 통과(H1~H5·H7·H8) — Qwen 1.5B 2.88GiB를 1GiB·512MiB, 3B 5.75GiB를 1GiB 예산에서 3회 통계 완주, 스왑 증가 0, 모든 방법 결과 비트 동일, 수동 캐시 우회 스트리밍의 0.95~1.03배 시간, 재사용 주기 정책이 회마다 예산만큼 붙잡아 LRU·스트리밍보다 11~22% 덜 읽음, 파생 512MiB 압축(2.91배) 경우도 256MiB에서 완주. 속도 이득은 없었음(계산 주도, 입출력 미중첩) → 2단계 미리 읽기 | 확정 |
| [0114](0114-e026-preregistration.md) | 2026-09-30 | experiment (사전 등록) + design | E026 = 관문 G-R2: 2단계 범위(미리 읽기 서비스 스레드, `derive` 재계산, 느려짐 예측, `memopro.rt.torch` 가중치 스트리밍과 역전파 참조 저장; 학습은 베이스 고정 LoRA). P1 미리 읽기 ≤ 0.95× 수동 스트리밍, L1 재계산 128MiB, C0 GPT-2 비트 동일, C1 1.5B 768MiB·1GiB, C2 3B 1.5GiB·1GiB(RSS ≤ 예산+256MiB, 스왑 ≤ 64MiB), R1 예측 ±25%, D0 GPT-2 LoRA 비트 동일, D1 1.5B LoRA 768MiB·1GiB, S1 OS 페이징 보고(스왑 감시자). 실행 전 수정 8건(GPT-2 예산 W/2, 정렬 참조, R1 시점, 두 번째 예산 1GiB, 오염 규칙, 비교 명확화, 드롭아웃 끔, 학습 데이터) | 확정 |
| [0115](0115-runtime-phase2-build.md) | 2026-09-30 | implementation | 런타임 C-R 2단계: 서비스 스레드가 배운 순서로 미리 읽기(나중에 쓸 것만 내보냄, 창 ≤ 한도/4), `derive` 재계산(다이제스트 확인, 입력 보호), `predict`(한 주기 기록의 Belady 모의), 비용 추정을 바이트·초 감쇠 합으로(예측 오차 5배 → 8%), `memopro.rt.torch`(`stream_model`: meta 골격 + safetensors 구간을 복사 없는 `frombuffer` 가중치로, `saved_weights`: 역전파가 가중치 대신 참조 저장). 발견: HF 매핑 가중치의 64바이트 어긋남이 BLAS 결과를 바꿈, 텐서 하나가 한도에 들어가야 함, M1 bf16 역전파 2.8 GFLOPS 경로 | 확정 |

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
