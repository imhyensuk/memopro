# memopro 아키텍처 설계

- **버전**: 설계 v0.4.0 (2026-09-26) — v0.2·v0.3 구현 반영: 품질 4등급·선호 기본 speed, 메타데이터 기반 크기, `check`의 할당 없는 추적(스크립트는 `run --dry-run`), `train_session`의 OOM 재시도·옵티마이저 교체는 제안만, census 정밀 모드 정의, γ는 안전 지점에서만, `run`은 필요할 때만 개입 ([0052](../research/0052-build-v02-v03-decision.md)·[0053](../research/0053-build-v02-v03.md)) / v0.3.11 (2026-09-25) — v0.1 구현 반영: 정체성 유지 해제(B1), 텐서별 방법 혼합, 자동 복원 hook, 엔진·codec 확정 (0036~0040) / v0.3.10 — 보수적 가용 메모리 정의, 예산 규칙 구현 반영 (0035) / v0.3.9 — 패키지 구조를 실제 뼈대에 맞춤, 오류 계층·설정 계층·지연 import 명시 (0034) / v0.3.8 — G3 장기 과제 명시, 실험 규칙 K1~K6 원칙화 (0033) / v0.3.7 — β SSD 쓰기 최소화(원본 재읽기 우선, SSD 기본 끔)·방법 선택 인터페이스·명시 핸들, census 재정의와 정밀 경량판 v0.1 (0032) / v0.3.6 — 중복성 검사 3차 반영: census 정밀 모드 재정의, 방출 엔진의 존재 이유와 기준선 명시(0029·0030) / v0.3.5: Rust 코어 설계 규칙 RS1~RS5 채택(0027) / v0.3.4: β 방출 기본값 무압축(0025 E008 C5) / v0.3.3: α 기각(0018), β 기본 모드 변경(0019)
- **이력**: v0.1 기법 모듈 중심(0008) → v0.2 범용성 2계층(0010) → v0.2.1 논리 수정(0011) → v0.3 대상·범위 조정(0012) → v0.3.1 검증 수정(0013) → v0.3.2 P1~P4 채택(0015) → v0.3.3 X1 결과 반영(0021)
- **근거 기록**: [0006](../research/0006-novel-technique-exploration.md), [0009](../research/0009-use-case-analysis.md), [0010](../research/0010-universality-redesign.md), [0011](../research/0011-design-v02-verification.md), [0012](../research/0012-revision-v03.md), [0013](../research/0013-revision-v03-verification.md)

---

## 0. 궁극적 목표

> **메모리가 부족한 PyTorch 개발자 누구나, 자기 프로젝트·서비스·개발·학습에서
> 큰 메모리를 요구하는 모델을 쉽게 쓸 수 있게 한다.**

**대상 (0012 S1)**: 하드웨어 사양이나 숙련도와 무관하게, PyTorch로 무언가를 만드는 사람.
직접 만든 모델, 확산·비전·오디오 모델, Python 코드 안의 LLM(HF Transformers), 파인튜닝, 연구 코드, Python 기반 서비스(FastAPI·Gradio 등)가 포함된다.
**대상 아님**: 코드 없이 LLM 앱을 쓰려는 최종 사용자. memopro는 이들을 외부 도구로 안내하지도 않는다(0012, S2 기각).

**목표별 범위 (0033)**: v0.1~v0.3은 **G1(전 단계 메모리 부담 최소화)·G2(더 작은 메모리 환경)** 중심이다. **G3(하드웨어 병목 극복)는 장기 과제**이며, 이를 직접 겨냥하는 기법은 γ뿐이다. G3 후보는 센서스 측정 연구(논문 A)의 결과에서 0006 절차(선행 조사 → 가설 → 사전 등록 실험 → 관문)로 탐색한다. 찾지 못하면 그대로 기록한다.

이 목표를 기준으로 삼으면 "새로운 기법 하나"만으로는 부족하다. 사용자는 기법이 아니라 **결과**(내 기기에서 돌아간다)를 원한다.
그래서 memopro는 두 계층으로 구성한다.

| 계층 | 역할 | 가치 | 원칙 |
|---|---|---|---|
| **범용 접근 계층** | 누구나 한 줄로 쓰게 한다. 환경을 감지하고, 예산을 산정하고, 기법을 골라 적용하고, 결과를 설명한다 | 실용성, 범용성 | 기존 검증된 기법은 **재구현하지 않고 선택적 백엔드로 연결** |
| **연구 코어** | memopro 고유 기법 (β hibernate, census, γ elastic. ~~α rfc~~ — 0018 기각) | 신규성, 학술성 | 선행 사례 조사와 검증 관문을 통과한 것만 |

## 1. 설계 원칙

| # | 원칙 | 의미 |
|---|---|---|
| U1 | **아무 설정 없이도 동작한다** | 기본값 `budget="auto"`, `quality="balanced"`. 사용자가 메모리, 양자화, 오프로드를 몰라도 된다. |
| U2 | **점진적 공개** | 초보자는 한 줄로 쓰고, 숙련자는 손잡이(옵션)를 쓰고, 전문가는 개별 기법과 Rust 코어를 직접 쓴다 (§4). |
| U3 | **memopro 자신의 오류로는 멈추지 않는다 (fail-open) + OOM 사전 예방** | 기법 적용이 실패하면 다음 구성이나 원래 경로로 계속 실행한다. 지원하지 않는 환경에서는 아무것도 하지 않고 이유를 알려준다. 단, OS의 강제 종료(Linux OOM killer, macOS jetsam)는 잡을 수 없으므로 **예산 여유분과 압박 감시로 사전에 예방**한다 (0011 L2). |
| U4 | **약속한 품질 이상으로 손실을 내지 않는다** | 사용자가 고른 `quality` 등급을 넘는 손실 기법은 자동 적용하지 않는다. |
| U5 | **충실도 3분류를 지킨다** | **정확**(수학적으로 동등) → 자동 / **수치 변경**(bf16, 8비트 옵티마이저, 양자화 등 — 품질 등급으로 선언) → 사용자가 고른 `quality` 범위 안에서만 자동 / **의미 변경**(LoRA 전환 등 학습 대상·구조 변경) → **제안만** (0011 L6) |
| U6 | **정직한 기대치** | 메모리 예상치(검증 목표 ±15%)와 품질 등급을 알려준다. 속도는 **대략적 추정(자릿수 수준)**임을 명시하고 모형을 공개한다: 디코드 ≈ 토큰당 읽는 바이트 / 유효 대역폭 (0011 L8). memopro가 제공하는 구성만 제시한다(외부 도구 안내 없음, 0012). |
| U7 | **환경을 정확히 안다** | 물리 메모리가 아니라 **실제 사용 가능한 메모리**를 기준으로 한다: 컨테이너(cgroup) 제한, 통합 메모리, GPU 여유분, 다른 앱의 사용량. |
| U8 | **설치가 쉽다** | `pip install memopro`만으로 동작한다(주요 플랫폼 사전 빌드 wheel). 무거운 의존성은 선택 설치(extras). |
| U9 | **확장 가능하다** | 모든 기법은 같은 `Technique` 인터페이스를 따른다. 제3자도 기법을 추가할 수 있다. |
| P1–P7 | (설계 v0.1 원칙 유지) | Rust는 프레임워크를 모름, 뜨거운 경로는 장치/차가운 경로는 Rust, 스스로 오차 측정과 폴백, 절감량 보고, 재현성 |
| K1 | **이상치 분리** (0033, 0018 E002) | 손실 저장(양자화 등)은 이상치 채널·토큰을 따로 처리한다. 블록 단위 최댓값 기준만으로는 오차 22% |
| K2 | **LayerNorm 관련 텐서 보호** (0018 E003) | 손실 기능은 LayerNorm 관련 저장 텐서를 보호하거나 제외한다. 게인 그래디언트가 가장 민감하다 |
| K3 | **최악값 보고** (0018 E003) | 전체 평균 지표만 보지 않고 텐서별 최악값을 보고한다 |
| K4 | **독립 복원** (0018 E002) | 서로 의존하는 복원은 오차가 누적된다. 복원은 서로 독립적이어야 한다 |
| K5 | **통합 메모리 상한** (S1 측정) | 통합 메모리의 장치 예산은 MPS 권장 한도(8GB 기기에서 5.33GB)를 상한으로 쓴다 |
| K6 | **외부 내부 의존 테스트** (transformers 5.x 인과 마스크) | 외부 라이브러리 내부 동작에 의존하는 기능은 버전별 동작 검증 테스트를 둔다 |
| P4 예외 | 전역 패치는 L0 명시 선택 시에만 | `memopro run`의 `from_pretrained` 로딩 정책 패치 등은 사용자가 L0 모드를 명시적으로 선택했을 때만 적용하고, 문서화하며, `report()`에 표시한다 (0011 L9) |

## 2. 전체 구조

```
┌─────────────────────────────── ① 진입점 (누구나) ───────────────────────────────┐
│ CLI              Python 한 줄            노트북          생태계 통합             설정 파일     │
│ memopro doctor   memopro.load()          %load_ext       HF Transformers·PEFT·  memopro.toml │
│ memopro check    memopro.optimize()      memopro         TRL·Diffusers·         MEMOPRO_*    │
│ memopro run      memopro.train_session()                 Lightning 콜백          환경변수      │
└──────────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────── ② 오케스트레이터 ────────────────────────────────────┐
│ 환경 감지 → 예산 산정 → 전략 선택(후보 구성) → 적용 → 감시(γ) → 보고(report)            │
│ (HW·OS·cgroup·     (풀별 예산        (품질·속도 선호,     (fail-open)  (절감량·속도·   │
│  프레임워크·백엔드)   벡터)             census 실측 반영)                 품질·폴백 내역) │
└──────────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────── ③ 기법 레지스트리 (Technique 인터페이스) ─────────────────┐
│  연구 코어 (memopro 고유)                  │  기존 기법 연동 (선택적 백엔드, 재구현 금지)       │
│  (α rfc — 0018 실험으로 기각)              │  가중치 양자화: bitsandbytes / torchao / HQQ     │
│  β hibernate 유휴 텐서 동면                 │  지연 로딩: safetensors mmap                    │
│  γ elastic OS 압박 탄력 런타임               │  오프로드: accelerate (CPU·디스크)               │
│  census 메모리 정보 센서스                   │  체크포인팅: torch.utils.checkpoint             │
│                                           │  8비트 옵티마이저: bitsandbytes / torchao        │
│                                           │  KV 캐시 양자화: transformers 내장 등            │
└──────────────────────────────────────┬─────────────────────────────────────────────┘
┌──────────────────────────── ④ Rust 코어 (crates.io `memopro`) ──────────────────────┐
│ hwinfo (메모리·cgroup·스왑·디스크 속도) · pressure (macOS·Linux PSI) · codec · ledger ·   │
│ spill · policy · census 통계 커널                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. 오케스트레이터

### 3.1 환경 감지 (`memopro.env`)
- **메모리**: 물리 RAM, 현재 가용 RAM, 스왑, **cgroup v1/v2 한도**(Docker·Kubernetes), 통합 메모리 여부(Apple Silicon), GPU별 여유 메모리(CUDA·MPS)
  - **가용 RAM은 보수적으로 정의한다 (0035)**: `total − used`. used는 압축이나 스왑 없이는 내줄 수 없는 메모리이다. macOS는 앱(익명) − 퍼저블 + wired + 압축기(Activity Monitor 기준), Linux는 `MemTotal − MemAvailable`이다. macOS 자체 추정값(active 익명 페이지 포함)은 참고용 `kernel_available_bytes`로만 보고한다. 8GB M1 실측: 1.55 GiB 대 3.27 GiB(스왑 4.9GB 사용 중).
- **장치**: CPU / MPS / CUDA (ROCm·XPU는 감지만 하고 미지원 시 fail-open)
- **소프트웨어**: torch 버전, 설치된 백엔드(bitsandbytes, torchao, HQQ, accelerate, peft)와 **현재 장치에서의 동작 가능 여부**
- **디스크**: SSD 여유 공간, 읽기 속도 (오프로드·방출 계획용)

### 3.2 예산 산정: 풀별 예산 벡터 (0011 L1 수정)

메모리는 하나의 숫자가 아니다. 분리형 GPU에서는 VRAM과 RAM이 **별개의 풀**이고, 오프로드는 한 풀의 부족분을 다른 풀로 넘기는 것이다.

```
Budget = { device: 장치 메모리 예산,  host: 호스트 RAM 예산,  disk: 방출·오프로드용 디스크 예산 }
```

| 환경 | device | host | disk |
|---|---|---|---|
| 분리형 GPU (CUDA) | GPU 여유 메모리(`mem_get_info`) × (1 − 여유분) | min(가용 RAM, 컨테이너 여유) × (1 − 여유분) | 여유 공간 − 전체 × `min_free_disk_fraction`(기본 20%, 0032 H4) |
| 통합 메모리 (Apple Silicon, MPS) | **host와 같은 풀**. min(MPS 권장 한도 − 드라이버 할당량, host 사용 가능량) × (1 − 여유분) (K5) | min(가용 RAM, 컨테이너 여유) × (1 − 여유분). CPU 텐서는 MPS 한도를 받지 않는다 | 위와 같음 |
| CPU 전용 | — | min(가용 RAM, 컨테이너 여유) × (1 − 여유분) | 위와 같음 |

- 구현: `memopro.orchestrator.budget.compute_budget` (0035). 디스크에 쓸 수 있는지는 예산이 아니라 `disk_writes` 정책이 정한다.

- 여유분(`headroom`) 기본값 10%(초기값, 실측으로 보정 예정). 사용자는 `budget="6GB"`로 device와 host를 함께 제한할 수 있다. 풀별 지정(`{"device": …, "host": …}`, `device=4GB,host=6GB`)과 비율 지정(`0.7`, `"70%"`)도 된다(0053). 설정은 측정한 예산보다 크게 만들지 않는다.
- γ가 켜져 있고 OS가 압박을 알리면 `load`의 예산은 경고 ×0.5, 위험 ×0.25가 된다(초기값, E013에서 보정).
- 통합 메모리에서는 "CPU 오프로드"가 메모리를 줄이지 못한다(같은 풀). 후보 구성에서 자동으로 제외한다.

### 3.3 전략 선택: 후보 구성 목록 (0011 L5 수정 — 선형 "사다리"에서 변경)

복잡한 최적화기(0004의 플래너)를 만들지 않는다. 대신 **완결된 기법 조합(구성)의 짧은 후보 목록**을 만든다.
각 구성을 사용자 선호(`prefer="quality"｜"speed"｜"memory"`)에 따른 비용으로 정렬하고, 예산 벡터를 만족하는 첫 구성을 고른다.
int8과 int4처럼 서로 **대체** 관계인 기법은 같은 구성에 함께 들어가지 않는다.
메모리 효과는 빠른 추정(파라미터 수 × 바이트)으로 거르고, 필요하면 census 실측(meta 장치 → 작은 배치 시험 실행 후 외삽)으로 확인한다.
시험 실행 자체가 OOM을 낼 수 있으므로, 작은 배치에서 시작해 외삽한다.

아래 표는 후보 구성을 이루는 **기법 목록**이다. 표의 순서는 `prefer="quality"`일 때의 대략적 선호 순서이다.

**추론 기법 목록** (`quality` 등급이 허용하는 것만)

| # | 기법 | 품질 | 속도 | 출처 |
|---|---|---|---|---|
| 0 | 지연 로딩(mmap) + 로딩 피크 제거 | 무손실 | 동일 | 기존 연동 |
| 1 | fp16/bf16 | 사실상 무손실 | 동일 이상 | 기존 연동 |
| 2 | int8 가중치 | 미세 손실 | 약간 느림 | 기존 연동 |
| 3 | KV 캐시 양자화 | 미세 손실 | 비슷함 | 기존 연동 |
| 4 | int4 가중치 | 작은 손실 | 비슷함 | 기존 연동 |
| 5 | CPU 오프로드 | 무손실 | 느림 | 기존 연동 |
| 6 | 디스크 오프로드 | 무손실 | 매우 느림 | 기존 연동 |
| + | γ 탄력 모드: 실행 중 메모리 압박에 따라 구성을 바꿈 | — | — | **memopro 고유** |

- 5(CPU 오프로드)는 분리형 GPU에서만 의미가 있다(통합 메모리에서는 같은 풀). 6(디스크 오프로드)은 모든 환경에서 유효하다(디스크는 별개 풀).
- 기법 2·4(int8·int4)는 대체 관계이다.

**학습 기법 목록** (충실도 3분류 U5 적용)

| # | 기법 | 충실도 | 자동 적용 | 출처 |
|---|---|---|---|---|
| 0 | 혼합 정밀도(bf16) | 수치 변경 | quality 범위 내 | 기존 연동 |
| 1 | 활성값 체크포인팅 | **정확** | ✅ | 기존 연동 |
| 2 | 활성값 오프로드 (분리형 GPU 전용) | 정확 | ✅ | 기존 연동 (Unsloth 방식) |
| ~~3~~ | ~~α 잔차 고정점 체크포인팅~~ — **0018에서 기각** (교정이 오차를 키움) | — | — | — |
| 4 | 8비트 옵티마이저 | 수치 변경 | ❌ 제안만 (사용자의 옵티마이저 객체를 바꾸게 되므로, 0052 E3) | 기존 연동 |
| 5 | 마이크로배치 + 그래디언트 누적 | 정확, 단 **BatchNorm·손실 정규화 감지 시 경고** | ✅ (조건부) | 기존 연동 |
| 6 | 옵티마이저 상태 CPU 오프로드 (분리형 GPU 전용) | 정확 | ❌ 제안만 (같은 이유, 0052 E3) | 기존 연동 |
| 7 | LoRA / QLoRA 전환 | **의미 변경** | ❌ 제안만 | 기존 연동 |

**구현된 학습 순서 (0053)**: 메모리가 부족하면 마이크로배치 절반 → 활성값 체크포인팅 → 활성값 오프로드(CUDA) → 혼합 정밀도(품질 허용 시, float16이면 항상 손실 스케일링). `s.step(batch, loss_fn)`은 OOM이 나면 그 배치의 그래디언트를 버리고 다음 단계로 처음부터 다시 한다. 마이크로배치는 표본 비율로 손실을 가중해 정확하다(`reduction="mean"|"sum"`).

**품질 등급과 선호 (0052 E3)**: `quality`는 자동 적용할 손실의 상한이다 — `"lossless"`(정확만) < `"high"`(+반정밀도) < `"balanced"`(기본, +int8) < `"low"`(+int4). `prefer`는 남은 후보의 순서다 — `"speed"`(기본, 위 추론 표의 순서), `"quality"`(손실 없는 오프로드를 양자화보다 먼저), `"memory"`(작은 것부터). 크기는 불러오기 전에 메타데이터(safetensors 헤더, HF 메타데이터 API, config의 meta 모델)로 계산한다.

**활성값 병목 (0009 관찰, 유효)**: LoRA·QLoRA 이후 남는 활성값 병목은 기존 기법 #1(체크포인팅)과 #2(오프로드)로 대응한다. α로 줄이려던 계획은 0018에서 기각되었다.

**개발 세션** (노트북·REPL): β 동면 + census 경고 ("이 셀 이후 메모리 3.2GB 증가, 유휴 텐서 2.1GB 회수 가능")

### 3.4 적용과 fail-open (0011 L2·L3 수정)

기법은 적용 시점에 따라 두 종류로 나눈다.

| 종류 | 예 | 실패·변경 시 |
|---|---|---|
| **로드 시점형** | 가중치 양자화, 지연 로딩, 장치 배치 | 로드 **전에** 구성을 결정한다. 실패하면 원본에서 **다음 구성으로 다시 로드**한다. 원본 사본을 메모리에 들고 있지 않는다(절감이 사라지므로) |
| **실행 시점형** | 체크포인팅, β, 마이크로배치, γ 조정 | `apply`/`revert`로 되돌릴 수 있다 |

- 적용 중 예외가 나면 해당 기법만 포기하고 다음 구성이나 원래 경로로 계속 진행한다. 모든 과정은 `report()`에 기록한다.
- 잡을 수 있는 OOM(`torch.OutOfMemoryError`, MPS 할당 실패)은 구성을 한 단계 내려 재시도한다.
- 잡을 수 없는 종료(OS 강제 종료)와 macOS의 스왑 폭주는 **사전 예방**으로 대응한다: 예산 여유분, 시험 실행 외삽, `pressure` 감시(γ).

### 3.5 보고 (`memopro.report()`)
사람이 읽는 요약과 기계가 읽는 JSON을 제공한다.
- 적용된 기법, 절감량, 예상·실측 속도 변화, 품질 등급, 폴백과 실패 내역
- 오류와 경고 메시지는 해결 방법을 함께 안내한다 (예: "bitsandbytes가 이 장치(MPS)를 지원하지 않아 torchao int8로 대체했습니다").

## 4. 점진적 공개: 사용 수준 5단계

| 수준 | 대상 | 방법 | 예 |
|---|---|---|---|
| **L0 코드 수정 없음** | 서비스 운영자, 비개발자 | CLI, 환경변수, 설정 파일 | `memopro run app.py --budget auto` |
| **L1 한 줄** | 대부분의 사용자 | `load`, `optimize`, `train_session`, `%load_ext` | `model = memopro.load("모델ID")` |
| **L2 손잡이** | 실무자 | `budget`, `quality`, `prefer="speed"｜"memory"`, `allow=[…]`, `deny=[…]` | `memopro.load(id, budget="6GB", quality="lossless")` |
| **L3 개별 기법** | 전문가, 연구자 | 기법 직접 호출 | `memopro.hibernate.now(obj, mode="spill")`, `memopro.census.record(..., mode="deep")` |
| **L4 확장·코어** | 라이브러리 제작자, Rust 개발자 | `Technique` 플러그인 작성, Rust 크레이트 | `@memopro.register_technique` / `use memopro::codec` |

### 4.1 L1 핵심 API (설계안)

```python
import memopro

# (1) 큰 모델을 내 기기에 맞춰 로드 (추론)                      [v0.2]
model, tok = memopro.load("meta-llama/Llama-3.1-8B-Instruct", task="text-generation")
#   → 환경 감지 → 예산 산정 → 추론 후보 구성 중 예산에 맞는 것을 선택 → 로드 → 요약 출력

# (2) 이미 가진 모델 최적화 (추론·학습 공용)                    [v0.2]
model = memopro.optimize(model, goal="infer")      # 또는 goal="train"

# (3) 학습 세션: 모델·옵티마이저·배치를 예산에 맞춤              [v0.2]
with memopro.train_session(model, optimizer, batch_size=32, budget="auto") as s:
    for batch in s.batches(loader):                 # 필요 시 마이크로배치 분할 + 누적 자동
        s.step(model(**batch).loss)

# (4) 진단·동면                                                [v0.1]
with memopro.census.record(model, optimizer) as c: ...   # 어디에, 얼마나 중복인가
%load_ext memopro                                        # 유휴 객체·회수 가능량 제안
%hibernate model_a                                       # β: 제안받은 객체를 명시적으로 동면
memopro.report()
```

### 4.2 L0: 코드 수정 없는 사용

```bash
memopro doctor                       # [v0.1] 내 환경: 풀별 예산 벡터, 장치, 설치된 백엔드
memopro check <모델ID 또는 폴더>      # [v0.2] 추론·학습 최대 메모리 예측(할당 없는 추적)과 memopro의 선택
memopro run app.py --budget auto     # [v0.3] 로딩 정책, γ, census(선택) 적용 후 실행. --dry-run은 계획만
```
- `memopro run`은 `from_pretrained` 로딩 정책, γ, census처럼 **모델 코드를 몰라도 되는 기능**만 자동 적용한다. 전역 패치는 이 모드에서만 허용한다(P4 예외). 로딩 정책은 스크립트가 배치·정밀도를 하나도 정하지 않았고 원래 형식으로는 들어가지 않을 때만 적용하며, 명시한 선택은 바꾸지 않는다(0052 E7).
- `check`는 스크립트를 받지 않는다. 스크립트에 대해서는 `memopro run --dry-run`이 계획을 보여 준다(0052 E3).
- 모델 구조가 필요한 기법(예: 학습 후보 기법)은 L1 이상에서 사용한다.

### 4.3 생태계 통합 (`memopro.integrations`)
| 대상 | 형태 |
|---|---|
| HF Transformers | `from_pretrained` 보조 함수, `Trainer` 콜백 |
| PEFT / TRL | 파인튜닝 레시피 콜백 (SFTTrainer 등) |
| Diffusers | 파이프라인 적합 로드 (이미지 생성 모델) |
| PyTorch Lightning | 콜백 |
| Jupyter / Colab | IPython 확장 (`%load_ext memopro`) |

통합 모듈은 해당 라이브러리가 설치된 경우에만 활성화한다(지연 import). 버전 호환 범위는 CI 매트릭스로 관리한다.

## 5. `Technique` 인터페이스 (연구 코어와 연동 기법 공통)

```python
class Technique(Protocol):
    name: str                       # "quant.int4.bnb", "hibernate", "census" ...
    stage: set[Stage]               # {INFER, TRAIN, DEV}
    quality: QualityGrade           # LOSSLESS | NEAR_LOSSLESS | SMALL_LOSS | ...
    fidelity: Fidelity              # EXACT | NUMERICS(품질 등급 선언) | SEMANTICS(제안만)
    timing: Timing                  # LOAD_TIME(재로드로 변경) | RUN_TIME(revert 가능)
    pools: set[Pool]                # 절감하는 풀: {DEVICE, HOST, DISK}
    origin: Origin                  # MEMOPRO_NATIVE | INTEGRATION

    def available(self, env: Env) -> Availability: ...     # 이 환경에서 가능한가 + 불가 사유
    def estimate(self, target, env: Env) -> Estimate: ...  # 예상 절감량·속도 비용
    def apply(self, target, env: Env) -> Applied: ...
    def revert(self, applied: Applied) -> None: ...        # RUN_TIME만. LOAD_TIME은 NotSupported → 재로드
    def report(self, applied: Applied) -> dict: ...
```

- 새 기법이 연구 코어에 들어오면(예: 센서스에서 발견한 새 기법) 후보 기법 하나로 등록된다. 구조를 바꿀 필요가 없다.
- 제3자 기법은 `entry_points` 그룹 `memopro.techniques`로 등록할 수 있다.

## 5.1 연구 코어 기법 설계 (설계 v0.1에서 이어짐, β는 0011·0013 수정)

### α. rfc (잔차 고정점 체크포인팅) — ⚠️ 0018에서 기각 (기록 보존용)

> E001~E003 결과: 사전학습 GPT-2의 블록은 모든 층에서 ρ > 1(비수축)이고, 역순 체인은 위층 오차를 교정 없이 누적한다. 교정 반복은 교정 없는 힌트(k=0)보다 그래디언트를 나쁘게 만들었다(4비트 k=3 코사인 0.786 대 k=0 0.997). 아래 설계는 기록으로만 남긴다.

```
순전파:  x_0 ─[블록0]→ x_1 ─[블록1]→ … → x_L
저장:    h_0=Q(x_0), h_1=Q(x_1), …, h_{L-1}   +  x_L (원래 정밀도)

역전파 (블록 l):  입력: x_{l+1} (블록 l+1의 역전파에서 이미 복원됨), h_l
   x ← dequant(h_l)
   반복 k회:  x ← x_{l+1} − f_l(x)         (옵션: Anderson 가속)
   잔차 r = ‖x + f_l(x) − x_{l+1}‖ / ‖x_{l+1}‖
   r > tol → 이 블록은 다음 스텝부터 원래 정밀도 체크포인트로 폴백 (원장에 기록)
   마지막 반복의 f_l(x) 계산 그래프로 역전파 (재계산 1회를 대체)
   복원한 x = x_l 을 블록 l−1에 전달 (RFCChain)
```

- 구현: 블록 단위 `autograd.Function` + 블록 간 상태 공유 `RFCChain`. 텐서 하나씩 처리하는 `saved_tensors_hooks`로는 블록 간 의존성을 표현하기 어렵다.
- 난수: dropout 등의 RNG 상태를 블록별로 저장했다가 반복마다 재생한다.
- 오프로드 결합 모드: 힌트 `h_l`을 호스트로 오프로드할 수 있다(분리형 GPU, 전송량 약 3.4배 감소).
- API: `memopro.rfc.wrap(model, hint_bits=4, max_iters=3, tol=1e-3, accel="anderson", verify=False)`

### β. hibernate (유휴 텐서 동면) — v0.1, 0011 L7 + 0013 V4~V8 + 0015 P3·P4 수정

**기본 동작: 제안 + 명시 실행 (0015 P4)**
- `%load_ext memopro`를 켜면 셀 실행 후 census가 **유휴 객체와 회수 가능량**을 알려준다.
  예: "`model_a` (nn.Module, 2.1GB, 7개 셀 동안 미사용) → `%hibernate model_a`로 회수 가능"
- 사용자가 `%hibernate <이름>` 또는 `memopro.hibernate.now(obj)`로 실행한다. 다시 쓰면 복원된다.
- **명시 핸들 방식 (0032 I5)**: `h = memopro.hibernate.now(obj)` → `obj = h.wake()`. 변수를 대리 객체로 바꿔 끼우지 않으므로 `isinstance`·`id()`·C 확장 문제가 없다. 대리 객체 방식(`%hibernate`)은 편의 기능이다.
- 자동 동면은 **선택 사항**이다: `memopro.hibernate.enable(auto=True, idle_cells=3)`. 자동 모드에서만 아래의 참조 그래프 규칙으로 대상을 스스로 고른다(명시 실행에도 같은 안전 검사를 적용한다).

**우선 환경과 기준선 (0015 P3)**

| 환경 | OS가 대신 처리하는가 | β의 위치 |
|---|---|---|
| CUDA GPU 메모리(VRAM) | ❌ 스왑하지 않음 | **1순위**: 유휴 VRAM 회수 |
| 스왑 없는 환경 (컨테이너 등, 환경별 확인 필요) | ❌ | **1순위**: OOM 강제 종료 방지 |
| Apple Silicon MPS 텐서 | 불확실 | 측정 후 결정 |
| CPU 텐서 (스왑 있는 환경) | ✅ 페이지 단위 압축·스왑 | **이득 측정 후에만 주장** (E005: 부동소수 인식 압축 대 OS 방식 압축) |

- 모든 β 평가는 다음 기준선과 비교한다: 아무것도 안 함 / 수동 `del` + `gc.collect()` + `empty_cache()` / OS 스왑·압축 / pytorch_memlab의 CPU 이동 기능.


**동면 대상 선정 (참조 그래프 기반)**
- 전역 `TorchDispatchMode`는 쓰지 않는다(모든 연산에 오버헤드).
- **해제 방식 (0036 B1로 수정)**: 텐서 객체는 두고 `tensor.data`만 크기 0 텐서로 바꾼다(복원 시 되돌림). 정체성이 유지되어 옵티마이저 등 참조가 복원 후에도 유효하다. 동면 중 직접 사용은 형상 오류로 크게 실패한다. 모듈은 forward pre-hook, 옵티마이저는 step pre-hook, 진행 중이던 역전파는 파라미터 그래디언트 hook으로 자동 복원한다. **다른 텐서와 저장공간을 공유하는 텐서(뷰, autograd가 저장한 텐서)와 meta 텐서는 동면하지 않는다**(0041 D1~D3: 교체하면 공유가 조용히 깨지거나 해제되지 않음). (이전 설계의 "`set_`으로 크기 0 저장공간을 만들지 않는다"는 이 방식으로 대체되었다.)
- 후보: 사용자 네임스페이스에 바인딩된 `torch.Tensor`와 `nn.Module` 중 N개 셀 또는 T초 동안 접근되지 않은 것
- `gc.get_referrers`로 외부 참조자를 찾는다.
  - 네임스페이스와 IPython 출력 캐시(`Out[n]`, `_`, `__`, `___`)의 참조는 "네임스페이스 참조"로 함께 집계한다. 동면 시 캐시 항목도 프록시로 교체한다 (V4).
  - 모델 파라미터를 참조하는 것이 **옵티마이저뿐**이면 모델 + 옵티마이저를 한 묶음으로 보고, **둘 다 유휴일 때만** 함께 동면한다 (V5).
  - 그 밖의 참조자(리스트, 다른 객체 속성 등)가 있으면 동면하지 않고 census 경고로만 알린다.
- 동면한 이름은 지연 프록시로 재바인딩한다. 프록시는 `__torch_function__`, 속성 접근, 호출 시 복원 후 원래 객체로 다시 바인딩한다.
  `isinstance`·`id()`·C 확장 직접 전달에는 한계가 있으며 문서화한다 (V8).

**동면 실행 (메모리를 늘리지 않게)**
- **청크 단위 스트리밍**: 장치 → 호스트 복사, 압축, 방출을 청크 단위로 처리해 순간 증가량에 상한을 둔다 (V6, RS4).

**방법 우선순위: SSD에 쓰지 않는 방법부터 (0032 H1)**

| 순서 | 방법 (`mode`) | SSD 쓰기 | 대상 | 결과 |
|---|---|---|---|---|
| 1 | **`source`**: 메모리만 해제, 복원 시 원본 파일(safetensors 등)에서 다시 읽음 | 없음 | 불러온 뒤 바뀌지 않은 텐서. **해시로 원본과 일치 확인**(텐서 버전 번호는 `.data` 경유 수정을 잡지 못함) | 원본과 동일 |
| 2 | **`host`**: GPU → 호스트 RAM 이동 | 없음 | CUDA 텐서(VRAM). 통합 메모리(MPS)에는 해당 없음 | 원본과 동일 |
| 3 | **`compress`**: RAM 안 무손실 압축 | 없음 | 압축이 잘 되는 텐서(저장 활성값 약 1.6배). fp32 가중치는 약 14%로 효과 작음(0019) | 원본과 동일 |
| — | **`bf16`**: 손실 압축 (**명시 선택만**, 자동 선택 안 함) | 없음 | 모든 장치. fp32 대비 절반(0019) | 수치 변경 |
| 4 | **`spill`**: SSD 기록 후 해제. 무압축 기본(0025 C5), 압축은 E009 뒤 재검토 | **있음** | 원본이 없는 데이터. **`disk_writes` 정책이 허락할 때만** | 원본과 동일 |

- 1MB 미만 객체는 제안 대상에서 제외한다.
- `mode="auto"`(기본)는 위 순서에서 가능한 첫 방법을 고르되 `bf16`은 고르지 않는다. 4번에 도달하면 정책에 따라 묻거나 거부한다.

**SSD 수명·성능 정책 (0032 H2~H4)**
- `disk_writes`: `"ask"`(기본, 쓰기 전에 크기를 알리고 확인) · `"never"`(금지, 어떤 명령보다 우선) · `"allow"`.
- 바뀌지 않은 데이터는 다시 쓰지 않는다: 복원 뒤에도 방출 파일을 디스크 예산 안에서 유지하고, 다음 동면 때 해시가 같으면 기존 파일을 재사용한다.
- 같은 객체를 짧은 시간 안에 반복해서 동면·복원하면 경고하고 RAM에 두기를 제안한다. 자동 모드와 γ에는 하루 쓰기 한도(`daily_write_limit`)를 둔다.
- 낮은 I/O 우선순위(macOS `setiopolicy_np`, Linux idle I/O 클래스 — NVMe 기본 스케줄러에서의 효과는 실측), SSD 여유 공간 20% 미만이면 `spill`을 쓰지 않음, 큰 순차 쓰기.
- 쓰기량은 `report()`·`%memopro status`·`doctor`에 오늘·누적으로 표시한다(H5).

**방출 파일 안전성 (0032 P5)**
- 권한 0600, 정상·비정상 종료 시 정리, 다음 실행 시 남은 파일 청소, 디스크 예산 준수.
- pickle을 쓰지 않는다. 원시 버퍼 + 메타데이터(dtype·shape·해시)로 저장한다.

**사용자 선택 인터페이스 (0032 H6)**
- 고른 방법이 불가능하면 조용히 바꾸지 않고 이유와 대안을 알린다. 예: "`source` 불가: 불러온 뒤 가중치가 수정됨 → `compress` 또는 `spill` 가능".
- 안전 검사(참조 그래프, 비트 단위 복원 확인)는 어떤 선택으로도 우회할 수 없다.
- `--plan`은 실행하지 않고 방법별 회수량·복원 시간(예상)·SSD 쓰기·결과 등급을 표로 보여 준다.

**회수 확인 (실측)**
- 해제 후 `torch.mps.empty_cache()` / `torch.cuda.empty_cache()`를 호출한다.
- 절감량은 논리 바이트가 아니라 **실측 회수량**(프로세스 RSS, `torch.mps.driver_allocated_memory()`, `torch.cuda.memory_reserved()`)으로 보고한다 (V7).

- API (0032 H6·I5):
  - 노트북: `%load_ext memopro` · `%hibernate <이름> [--mode auto｜source｜host｜compress｜bf16｜spill] [--plan]` · `%wake <이름>` · `%memopro status`
  - Python: `h = memopro.hibernate.now(obj, mode="auto")` → `h.wake()` · `memopro.hibernate.plan(obj)` · `memopro.hibernate.suggest()` · `memopro.hibernate.enable(auto=False, idle_cells=3, idle_seconds=None)`
  - 정책: `memopro.configure(hibernate_modes=[...], disk_writes="ask"｜"never"｜"allow", daily_write_limit="20GB", spill_dir=None)` · `memopro.toml` · `MEMOPRO_DISK_WRITES` · `memopro run --disk-writes never`

### census — v0.1, 0013 V9·V10 수정

| 모드 | 기본 | 측정 | 비용 |
|---|---|---|---|
| **빠른 모드** | ✅ | ① 범주별 바이트: 파라미터·그래디언트·옵티마이저 상태·역전파 저장 텐서(`saved_tensors_hooks`) ③ 이상치·거대 활성값 탐지, 생애 주기·유휴 시간, 셀 단위 증감(노트북). 무손실 엔트로피는 **보조 지표** (0032 R2) | 낮음 |
| **정밀 경량판** (v0.1, 0032 Q1) | 선택 | ② 정밀도 중복: 표본 텐서 몇 개에 대해 비트 폭 2·4·8에서의 오차·그래디언트 영향 → 범주별 "필요 비트" 추정 ④ 텐서별 최악 민감도(K3) | 중간 |
| **정밀 모드** (v0.2, 0015 P2, 구현 0053) | 선택 | 학습 상태 범주(가중치·그래디언트·옵티마이저 모멘트·저장 활성값)별로 **필요한 비트 대비 실제 저장 비트**를 측정한다. 범주 전체를 2·4·8·16비트로 교란해 손실 상대 변화·그래디언트 코사인·한 스텝 갱신량 코사인을 잰다(기준 1e-3, 0.999). `probe=`(고정 배치의 손실) 필요. 끝나면 모든 상태를 비트 단위로 복원 | 높음 (명시적 선택) |

- **커버리지 대조**: 할당자 통계(장치·호스트)와 비교해 분류하지 못한 부분을 **"미분류"**로 명시한다. `no_grad` 추론 텐서(KV 캐시 등), 할당자 캐시, 런타임 컨텍스트가 여기에 해당한다. 목표는 할당자 바이트의 90% 이상 분류이다 (V9).
- **차별점**: "어디에 쓰이나"는 기존 프로파일러(PyTorch Profiler, pytorch_memlab, memray, Scalene)와 겹친다. **"얼마나 중복(낭비)인가"**가 핵심 기능이다 (0013 D9·D10).
- **정밀 모드는 양자화 민감도 분석이 아니다 (0030 C3)**: 추론 모델의 층별 양자화 민감도는 AIMET QuantAnalyzer, HAWQ 계열, Intel Neural Compressor 등 기존 도구의 영역이다. memopro는 이를 재구현하지 않는다. 정밀 모드의 대상은 **학습 중 메모리 상태 전체**이고, 결과는 "어느 범주에 몇 비트가 낭비되는가"라는 메모리 조사로 보고한다.
- **행동 가능한 권고 (0032 P6)**: 결과를 조치로 연결한다. 예: "옵티마이저 상태는 원소당 약 k비트면 충분 → 8비트 옵티마이저 적용 시 약 X GB 절감 예상(수치 변경 등급)". memopro 기능이나 일반 기법 이름으로만 표현한다(외부 도구 안내 없음, S2).
- **정확성 검증 (0032 P7)**: 정보량을 아는 합성 텐서로 측정값을 확인하고, 할당자 통계와 대조한다.
- **통합 (0032 I8)**: HF Trainer 콜백, Lightning 콜백을 v0.1부터 제공한다.
- API: `with memopro.census.record(model, optimizer, mode="fast"｜"light"｜"deep", probe=…) as c: …` → `c.summary()`, `c.to_json()`

### γ. elastic (OS 압박 탄력 런타임) — v0.3, 0052 E6, 구현 0053

- **신호**: Rust `pressure::current()` — macOS `kern.memorystatus_vm_pressure_level`(1·2·4), Linux PSI(자기 cgroup v2의 `memory.pressure`, 없으면 `/proc/pressure/memory`의 some·full avg10; 경고 some ≥ 10 또는 full ≥ 1, 위험 some ≥ 40 또는 full ≥ 10, 초기값). 그 밖의 OS는 `Unsupported`(fail-open).
- **감시 스레드는 기록만 한다.** 사용 중일 수 있는 텐서를 백그라운드에서 바꾸면 사용자 계산이 깨지기 때문이다. 조치는 안전 지점에서만: 노트북 셀 경계(캐시 비우기 → 유휴 객체 쓰기 없는 동면, `spill`은 위험 수준 + `disk_writes="allow"`만), `train_session` 스텝 경계(수준당 한 단계씩 정확한 기법으로 내리고 `up_after`초 평온하면 한 단계 올림), `load` 예산(경고 ×0.5, 위험 ×0.25), `memopro.elastic.checkpoint()`.
- 선행 사례와 차이는 0051. 평가(E013)는 별도 사전 등록: 기준선은 자기 VRAM 되먹임(Tri-Accel식)과 OOM 뒤 재시도(accelerate).

## 6. Rust 코어 (`crates/memopro`) — crates.io

| 모듈 | 책임 | 비고 |
|---|---|---|
| **`hwinfo`** (v0.1, ✅ 0035) | 풀별 예산 벡터의 OS 수준 기반: 물리·가용(보수적)·OS 추정 RAM, 스왑, **cgroup v1/v2 한도**, CPU, 디스크 용량(`statvfs`) | `sysinfo` 0.39(system 기능만, cgroup 포함)와 `libc::statvfs`. MPS·CUDA 한도는 Python에서 torch로 조회 (0013 V12) |
| `pressure` (v0.3) | OS 메모리 압박 구독 (macOS memory pressure, Linux PSI) | γ. v0.1의 β는 유휴 기준으로만 동작 (0013 V14) |
| `codec` (v0.1, ✅ 0038) | 바이트 셔플 + **zstd 크레이트**로 부동소수 무손실 압축(`pack`/`unpack_into`, 스레드별 문맥 재사용) | β `compress`. **기존 크레이트 위에 얇게 구현하며 신규성 주장 없음** (0013 V11). E008 프로토타입 제거 |
| `ledger`, `spill` (v0.1, ✅ 0038), `policy` | 원장, **저장 엔진**: 원본 파일 → 목적지 버퍼 직접 읽기 + xxh3-128 다이제스트 확인(주 경로), 0600 방출 파일(최후 수단, 저우선순위 쓰기 풀), 중간 버퍼 없음(RS1~RS5) | β. **신규성 주장 없음**(0029 D20). 존재 이유: macOS 지원, 메모리 상한, β 결합 |
| `census` (v0.1) | 표본 엔트로피 등 통계 커널 | census |

> v0.1 크레이트는 기존 크레이트를 얇게 감싼 부분이 많아 단독 가치가 크지 않다. crates.io 배포는 최종 목표 달성과 이름 확보가 목적이며, 크레이트 설명에 범위를 명시한다 (0013 V13).

`crates/memopro-py`: PyO3 바인딩 (`publish = false`). E008용 프로토타입 바인딩(`codec_*`, `bytes` 입력)은 RS2 위반이며, 방출 엔진이 들어오면 공개 API에서 제거한다 (0027).

### 6.1 Rust 코어 설계 규칙 (0027에서 RS1~RS5 채택, RS6~RS8 보류)

성능을 만든 것은 언어가 아니라 설계였다(0025: 단순 Rust는 Python 스레드보다 느렸고, 설계한 Rust는 1.49배 빨랐다). 모든 Rust 대용량 경로는 다음 규칙을 따른다.

| # | 규칙 | 근거 | 완료 조건 (테스트) |
|---|---|---|---|
| RS1 | 방출·복원은 **하나의 Rust 호출** 안에서 끝난다. 데이터 크기만 한 Python 객체를 만들지 않는다. 복원은 미리 할당한 텐서 저장공간에 직접 쓴다 | E008 C4, E008b −18% | 방출·복원 중 Python 할당 최대치 < 청크 1개 |
| RS2 | 입력은 **buffer protocol로 빌린다**(`bytes` 금지). bf16 등은 `tensor.view(torch.uint8)` 뷰로 넘긴다 | 0013 V6, E008 C4 | 입력 주소 = `data_ptr()` |
| RS3 | zstd 문맥·작업 버퍼는 **스레드마다 한 번** 생성, 스레드 풀은 **프로세스에 하나** | E008b v1 → v2 2.1배 | 반복 호출 시 스레드·문맥 수 불변 |
| RS4 | 추가 메모리 ≤ **버퍼 수 × 청크 크기 + 상수**. 고정 크기 버퍼 풀 + 용량 제한 채널(백프레셔) | E008 C4 실패, E008b 367MiB | 입력 크기 2배에도 RSS 증가량 불변 |
| RS5 | 압축 ↔ 쓰기, 읽기 ↔ 해제를 **겹친다** (CUDA의 GPU → 호스트 복사 중첩은 외부 GPU 환경에서 추후) | E008b | 파이프라인 시간 < 순차 합 |

- 보류(0027): RS6 특화 커널·SIMD(기존 2·4바이트 셔플은 유지하되 요구하지 않음), RS7 GIL과 무관한 백그라운드 서비스(N3에서 재판단), RS8 `gil_used = false` 선언.
- **Rust를 쓰지 않는 곳**(0025): GPU 수학과 CPU 행렬곱(torch 담당, E007), C 라이브러리를 호출만 하는 단순 병렬(Python 스레드로 충분, E008 C3), 작은 호출이 잦은 경로(FFI 비용).

## 7. 패키지 구조 (0034 뼈대 기준)

```
python/memopro/
├── __init__.py          공개 API (PEP 562 지연 import), load_ipython_extension
├── _errors.py           MemoproError ← InvalidArgument(ValueError) ← ConfigError / PolicyError /
│                        ModeUnavailable(이유 + 대안) / NotYetImplemented(예정 버전 명시)
├── _units.py            크기 파싱·표시
├── config.py            기본값 < memopro.toml($MEMOPRO_CONFIG) < MEMOPRO_* < configure()
├── report.py            report() — 적용·실패·회수량(실측)·SSD 쓰기량
├── access/              load · optimize · train_session · check (v0.2): _info(메타데이터 크기) · _load ·
│                        _check(MemTracker+FakeTensorMode) · _optimize · _train · _common(설정·장치·예산)
├── _run.py              memopro run: from_pretrained 로딩 정책(run 모드만), γ, census
├── cli.py, __main__.py  memopro doctor | check | run  (종료 코드 0/1/2/3=미구현)
├── env/                 detect() → Env(HostMemory, Disk, Device…)
├── orchestrator/        budget.Budget · candidates.Configuration · apply.fail_open
├── hibernate/           now → Handle.wake · plan → PlanRow · wake · suggest · enable · status
│   └── _policy.py       resolve_modes: 쓰기 없는 방법 우선, bf16 명시만, SSD는 disk_writes 정책
├── census/              record(mode = fast | light | deep) → Census.summary / to_json; _deep(정밀 모드)
├── elastic/             γ (v0.3): enable · disable · status · current · checkpoint · 안전 지점 hook
├── integrations/        ipython(%hibernate · %wake · %memopro) · hf · lightning (census 콜백)
├── techniques/          base(Technique·Registry) · native/ · integrations/loading(로드 시점 기법·백엔드 시험)
└── _core.*.so           Rust 확장

crates/memopro/src/      error · hwinfo · spill(원본 재읽기·다이제스트·방출 파일, Unix·Windows) · ledger · pressure(macOS·Linux PSI) · codec
```

- **지연 import 규칙 (0032 I4)**: `import memopro`는 Rust 확장과 최상위 모듈만 불러온다. torch·numpy·transformers·IPython과 하위 모듈은 처음 쓸 때 불러온다. `tests/test_import_cost.py`가 보장한다.
- **미구현 규칙 (0034 K-a)**: 설계된 API 중 아직 없는 기능은 조용히 아무것도 하지 않지 않고 `NotYetImplemented`(Rust는 `Error::NotImplemented`)로 예정 버전을 알린다. Rust 코어는 패닉(`unimplemented!`)을 쓰지 않는다.

## 8. 설치와 플랫폼

| 설치 명령 | 포함 |
|---|---|
| `pip install memopro` | 코어 + CLI (`doctor`), numpy. torch 없이도 OS 수준 풀(RAM·cgroup·디스크) 진단 가능. MPS·CUDA 한도는 "torch 설치 필요"로 표시 (0013 V12) |
| `pip install memopro[torch]` | PyTorch 기능: v0.1 census·β, v0.2 load·optimize·train_session, 연구 코어 |
| `pip install memopro[hf]` | + transformers, accelerate, safetensors, peft |
| `pip install memopro[all]` | + 선택 백엔드 중 현재 플랫폼에서 설치 가능한 것 |

- 사전 빌드 wheel: macOS arm64/x86_64, Linux x86_64/aarch64 (manylinux), Windows x86_64
- 장치 지원 등급: **1급** CPU, MPS, CUDA / **감지만** ROCm, XPU (fail-open)
- 문서: 영어(기본, 국제 사용자) + 한국어

## 9. 품질 체계 (설계 v0.1에서 확장)

| 영역 | 추가 사항 |
|---|---|
| fail-open 테스트 | 모든 기법에 강제 예외를 주입해 원래 경로로 계속 실행되는지 확인 |
| 환경 매트릭스 | CI에서 CPU 전용, macOS MPS, (가능 시) CUDA 러너 · 컨테이너 메모리 제한 시나리오(cgroup) |
| 백엔드 계약 테스트 | 연동 백엔드별 최소·최신 버전에서 `available/apply/revert` 계약 확인 |
| 사용성 테스트 | "처음 쓰는 사람 시나리오" 예제 노트북이 CI에서 끝까지 실행되는지 확인 |

## 10. 미결정 사항

| 항목 | 권고 |
|---|---|
| 라이선스 | MIT OR Apache-2.0 |
| 최소 Python | **3.11** (abi3-py311) — buffer protocol로 텐서를 복사 없이 받기 위함 (0026) |
| GitHub 위치 | 사용자 결정 |
| 이름 선점 0.0.1 | 사용자 확인 후 |
| α 채택 | **0018에서 기각** (0021 D1) |
| 첫 연동 백엔드 범위 (v0.2) | 권고: safetensors, accelerate, torch checkpoint, torchao, bitsandbytes(CUDA·MPS — MPS는 torch ≥ 2.9 필요, 0011) |
