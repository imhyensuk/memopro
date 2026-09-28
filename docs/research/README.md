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
