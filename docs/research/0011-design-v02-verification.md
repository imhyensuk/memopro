# 0011. 설계 v0.2 검증: 중복성과 논리성

- **날짜**: 2026-09-24
- **유형**: survey + decision
- **상태**: 검증 확정 / 논리 오류 수정은 설계 v0.2.1에 반영 / 전략 제안: S1·S3 승인, S2 기각 (→ 0012)
- **관련 기록**: 0010 (검증 대상), 0006 (연구 코어 후보)

## 배경 / 동기

사용자 요청: "중복성과 논리성 검증을 우선 진행해."
설계 v0.2(0010)에서 새로 추가한 **범용 접근 계층**은 선행 조사를 하지 않았다. 설계 내부의 모순과 실현 가능성도 점검한 적이 없다.

## 방법

- 중복성: 접근 계층의 각 기능(`doctor`, `check`, `load`, `train_session`)과 연구 코어(α, β, census)에 대응하는 기존 도구를 웹에서 조사 (2026-09).
- 논리성: 설계 문서의 원칙, 공식, 인터페이스, 완료 조건을 하나씩 반례와 실현 가능성으로 점검.

## 결과 1 — 중복성

| # | memopro 구성 요소 | 기존 도구 | 중복도 | 판단 |
|---|---|---|---|---|
| D1 | `doctor` / `check` (하드웨어 감지, 모델 적합 판정, 양자화 추천, 속도 추정) | **llmfit** (Rust CLI, crates.io. 하드웨어 감지, 모델별 적합·품질·속도 점수, 동적 양자화 선택, macOS·Linux·Windows), **LM Studio** (`--estimate-only`, 적합 배지, 로드 전 경고), quantfit | 🔴 매우 높음 (LLM) | LLM 추론 한정으로는 이미 해결된 문제 |
| D2 | `load` (가용 메모리에 자동 적합 로드) | **llama.cpp `--fit`** (기본 켜짐, 여유 메모리에 맞춰 컨텍스트·GPU 층 수·텐서 분할 자동 조정), **Ollama** (VRAM 기반 층 오프로드 자동 계산), **HF `device_map="auto"` + `max_memory` + 양자화 설정** (PyTorch 경로) | 🔴 매우 높음 (LLM) | 최종 사용자의 LLM 채팅·서빙은 Ollama·LM Studio·llama.cpp가 "누구나" 수준으로 이미 제공 |
| D3 | `train_session` (학습 적합) | **Unsloth** (VRAM에 따라 청크 수 자동 조정, 활성값 비동기 RAM 오프로드 — 30% 추가 절감, +1.9% 시간), HF `auto_find_batch_size`, DeepSpeed Autotuning, torch `activation_memory_budget` | 🟠 높음 | LLM 파인튜닝 한정으로는 강한 경쟁자 존재 |
| D4 | **α** (활성값 체크포인트 절감) | **Unsloth 활성값 RAM 오프로드**가 같은 병목(LoRA 이후 활성값)을 낮은 오버헤드로 해결 | 🟠 목표 병목 중복 | 분리형 GPU + 넉넉한 호스트 RAM 환경에서는 오프로드가 α보다 유리. α의 고유 영역은 아래 L10 참조 |
| D5 | **β** (유휴 텐서 동면) | Dask·Ray: 메모리 압박 시 디스크 방출 + 투명 복원 (데이터 프레임워크 관리 객체 한정). **IPyExperiments**: Jupyter 셀 단위 메모리 추적, 실험 변수 자동 해제 | 🟡 개념 중복 | "임의의 PyTorch 텐서·모듈을 노트북에서 압축·방출·투명 복원"하는 사례는 여전히 미발견 |
| D6 | **census** | 메모리 위치 추적: PyTorch 메모리 스냅샷, IPyExperiments, 각종 프로파일러 | 🟡 부분 | "어디에 쓰이나"는 중복. "**얼마나 중복(낭비)인가**"(실제 엔트로피, 그래디언트 민감 비트)는 미발견 |
| D7 | γ | PagedWeight, Tri-Accel, eLLM (0006) | 🟡 중간 | 변동 없음 |

**종합**: 0010의 v0.1 계획(LLM 추론 접근: doctor, check, load)은 **이미 있는 도구의 재구현**에 가깝다.
이는 0010의 "재구현 금지" 원칙과도 충돌한다. llmfit과 llama.cpp `--fit`은 기능 단위로 겹친다.

## 결과 2 — 논리성

| # | 심각도 | 문제 | 반례·근거 | 수정 |
|---|---|---|---|---|
| L1 | 🔴 치명 | 예산 공식 `min(가용 RAM, cgroup, 장치 여유)`이 틀렸다 | 분리형 GPU에서 VRAM과 RAM은 **별개 풀**이다. min을 쓰면 오프로드 계획(RAM에 두는 부분)이 불가능해진다 | **풀별 예산 벡터** {장치, 호스트, 디스크}. 통합 메모리는 단일 풀이되 MPS 한도(`recommendedMaxWorkingSetSize` × 워터마크 비율)를 적용 |
| L2 | 🔴 치명 | "절대 사용자 프로그램을 죽이지 않는다"는 지킬 수 없는 약속 | Linux OOM killer, macOS jetsam은 SIGKILL로 프로세스를 종료한다(잡을 수 없음). macOS CPU 메모리 초과는 예외가 아니라 스왑 폭주로 나타난다 | "memopro **자신의** 오류로는 멈추지 않는다 + OOM을 **사전 예방**하고 압박을 감시한다"로 정정 |
| L3 | 🔴 치명 | `apply`/`revert` 계약과 메모리 절감이 모순 | 로드 시 양자화한 가중치는 원본을 보관해야 되돌릴 수 있다. 원본을 보관하면 절감이 사라진다 | 기법을 **로드 시점형**(로드 전에 구성 결정, 실패하면 다음 구성으로 재로드)과 **실행 시점형**(되돌리기 가능)으로 구분 |
| L4 | 🟠 높음 | v0.1 완료 조건 "M1 8GB에서 7~8B LLM을 `load` 한 줄로 실행"의 실현성과 타당성 | 8B int4 가중치만 약 4.5~5GB. 8GB M1의 MPS 한도와 OS·앱 점유를 고려하면 경계선이거나 오프로드가 필요하다(매우 느림). bitsandbytes MPS 4비트는 torch ≥ 2.9가 필요하고 최적 커널은 macOS 26 이상. torchao int4는 MPS에서 에뮬레이션이다. 이 용도에는 llama.cpp·MLX·Ollama가 훨씬 적합하다 | 완료 조건 재정의 필요 → 전략 제안 S2 |
| L5 | 🟠 높음 | "적합 사다리"를 선형 사다리로 정의할 수 없다 | int8과 int4는 누적이 아니라 **대체** 관계이다. `prefer="speed"`면 int4(빠름)가 CPU 오프로드(무손실, 느림)보다 앞서야 하므로 순서가 선호에 따라 바뀐다 | 사다리 → **후보 구성 목록**(각각 완결된 기법 조합) + 선호 기반 비용으로 정렬 |
| L6 | 🟠 높음 | "학습 의미 보존" 정의가 일관되지 않는다 | bf16 혼합 정밀도, 8비트 옵티마이저는 수치를 바꾼다. 마이크로배치 + 누적은 **BatchNorm 통계**와 **토큰 평균 손실의 정규화**를 바꿀 수 있다 | 3분류: **정확**(수학적으로 동등) / **수치 변경**(품질 등급으로 선언, 동의 범위 내 자동) / **의미 변경**(제안만). 마이크로배치는 BatchNorm과 손실 정규화를 감지해 경고 |
| L7 | 🟠 높음 | β 구현 방식의 결함 | 전역 `TorchDispatchMode`는 노트북의 **모든 연산**에 Python 수준 오버헤드를 준다. 저장공간을 크기 0으로 바꾸는(`set_`) 방식은 가로채기에서 빠진 경로가 있으면 **조용히 틀린 결과**를 낼 수 있다 | 전역 모드 금지. ① 다른 참조가 없는 네임스페이스 변수는 지연 프록시로 재바인딩 ② `nn.Module`은 모듈 단위로 동면하고 `forward_pre_hook`으로 복원 ③ 그 외에는 동면하지 않음 |
| L8 | 🟡 중간 | U6(예상 속도 제시)가 정확도 근거 없이 약속됨 | 메모리 예측은 검증 가능하지만(±15%), 속도는 커널·대역폭·오프로드 I/O에 좌우된다 | 속도는 **대략적 추정**(자릿수 수준)으로 명시하고 모형을 공개: 디코드 ≈ 토큰당 읽는 바이트 / 유효 대역폭 |
| L9 | 🟡 중간 | `memopro run`이 `from_pretrained`를 패치하는 것은 P4(전역 몽키패칭 금지)와 충돌 | — | P4 예외 조항: L0 명시적 선택 시에만, 문서화하고 `report`에 표시 |
| L10 | 🟡 중간 | α의 사다리 위치가 환경을 고려하지 않음 | 분리형 GPU + 넉넉한 RAM이면 활성값 오프로드(Unsloth 방식)가 오버헤드(+1.9%) 면에서 α(추정 +50%)보다 낫다 | α의 고유 영역: **통합 메모리**(오프로드할 "다른 풀"이 없음), **호스트 RAM도 부족한 환경**(Colab 12GB 등), **오프로드와 결합**(4비트 힌트를 오프로드하면 전송량이 약 3.4배 감소). 후보 구성 정렬에 환경 조건 반영, E3 기준선에 활성값 오프로드 추가 |
| L11 | 🟡 중간 | "누구나 큰 모델 서비스"의 대상이 모호 | LLM 최종 사용자는 Ollama·LM Studio가 이미 잘 지원한다 | 대상 재정의 → 전략 제안 S1 |
| L12 | 🟢 낮음 | G3(하드웨어 병목) 대응이 약함 | 접근 계층은 대역폭 문제를 다루지 않는다 | 알려진 한계로 유지 (α: 메모리 → 연산 치환, γ) |
| L13 | 🟢 낮음 | Rust `hwinfo`가 기존 크레이트를 재구현할 위험 | sysinfo 등 기존 크레이트가 있다 | 기존 크레이트를 의존성으로 쓰고, cgroup·MPS 한도 등 빠진 부분만 구현 |
| L14 | 🟢 낮음 | 문서 간 표기 불일치 | use-cases ①은 `%load_ext memopro.hibernate`, 나머지 문서는 `%load_ext memopro` | `%load_ext memopro`로 통일 |

## 전략 제안 (사용자 승인 필요)

**S1. 대상 재정의: "LLM 최종 사용자"가 아니라 "PyTorch로 자기 프로젝트·서비스를 만드는 개발자"**
- 사용자 원문("자신의 프로젝트, 서비스, 개발, 학습, 사용")과 일치한다.
- Ollama·llama.cpp를 쓸 수 없는 영역을 겨냥한다: 직접 만든 모델, 확산(이미지 생성) 파이프라인, 비전·오디오 모델, 파인튜닝, 연구 코드, Python 기반 서비스.

**S2. 재구현 대신 "안내(라우팅)"**
- `memopro check`가 용도에 따라 **가장 적합한 기존 도구를 추천**한다. 예: LLM 채팅이면 "Ollama/llama.cpp로 Q4_K_M 권장" + 명령어 제공.
- 경쟁 대신 안내하는 것이 "누구나"에 더 부합하고, 재구현 금지 원칙과도 일치한다. v0.1의 LLM `load` 완료 조건은 폐기한다.

**S3. v0.1 재구성: 고유성이 확인된 것부터**
- `doctor`(풀별 예산 벡터, MPS·cgroup 한도, 경쟁 도구와 차별되는 기반) + **census**(메모리 중복도 보고 — 고유) + **β**(노트북 동면 — 고유 설정)
- PyTorch 범용 `optimize`와 `train_session`은 v0.2로 옮긴다. α는 관문 통과 시 합류한다.

## 논문 매핑

- **Related Work**: 결과 1 표(llmfit, llama.cpp fit, Ollama, LM Studio, Unsloth, Dask·Ray 방출, IPyExperiments).
- **Method**: L1(풀별 예산), L3(로드 시점형·실행 시점형), L6(정확·수치 변경·의미 변경 3분류)는 시스템 설계의 정식 정의로 쓸 수 있다.
- **Discussion**: α의 고유 영역(L10) — 통합 메모리 시대(Apple Silicon, 통합 메모리 PC·워크스테이션)의 활성값 문제.

## 출처 (URL)

- llmfit: https://www.llmfit.org/ , https://lib.rs/crates/llmfit , https://github.com/AlexsJones/llmfit · quantfit: https://github.com/Sahil170595/quantfit
- llama.cpp fit-params: https://github.com/ggml-org/llama.cpp/tree/master/tools/fit-params , https://deepwiki.com/ggml-org/llama.cpp/3.11-memory-optimization-and-llama_params_fit
- Ollama 자동 오프로드: https://github.com/ggml-org/llama.cpp/discussions/4049 , https://eastondev.com/blog/en/posts/ai/ollama-gpu-scheduling/
- LM Studio: https://lmstudio.ai/docs/cli/local-models/load , https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1631
- HF device_map/max_memory: https://huggingface.co/docs/transformers/v4.34.1/main_classes/quantization
- Unsloth: https://unsloth.ai/blog/long-context , https://unsloth.ai/docs/blog/500k-context-length-fine-tuning
- Dask: https://distributed.dask.org/en/stable/worker-memory.html · Ray: https://docs.ray.io/en/latest/ray-core/objects/object-spilling.html
- IPyExperiments: https://github.com/stas00/ipyexperiments
- bitsandbytes MPS: https://github.com/bitsandbytes-foundation/bitsandbytes/releases , https://huggingface.co/docs/bitsandbytes/main/en/installation · torchao: https://github.com/pytorch/ao
- PyTorch MPS 메모리: https://docs.pytorch.org/docs/2.8/mps_environment_variables.html , https://docs.pytorch.org/docs/stable/generated/torch.mps.set_per_process_memory_fraction.html
