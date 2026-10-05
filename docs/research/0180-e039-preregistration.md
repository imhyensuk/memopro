# 0180. E039 사전 등록: 흘려 쓰는 무손실 추론 비교 — memopro 대 llama.cpp(mmap)·AirLLM, 8GB M1, Qwen2.5-3B

- **날짜**: 2026-10-05
- **유형**: experiment (사전 등록)
- **상태**: 확정. 실행 전
- **관련 기록**: 0179 (채점표 S8·S9), 0152 (E033c, memopro 3B 생성), 0151 (E034 비교 방식), 0125 (AirLLM 조사)

## 사용자 지시

> "설치 허락할게, CI 통과하면 병합하고 1~3번 순서대로 진행해." 1번 = 0179의 "흘려 쓰는 추론 비교: llama.cpp(mmap)·AirLLM".

## 설치·준비 (측정 전, 모두 프로젝트의 `.cache/` 안, git 무시)

| 항목 | 내용 |
|---|---|
| llama.cpp | Homebrew `llama.cpp` 0.5.0 (build 11146, commit `7fe450e19`) |
| 변환 | 같은 커밋의 `convert_hf_to_gguf.py`와 `gguf-py`(`.cache/llama.cpp-src`, 패키지는 `.cache/gguf-pkgs`, sentencepiece 포함). 로컬 Qwen2.5-3B-Instruct safetensors를 `--outtype bf16`으로 변환 → `.cache/gguf/qwen2.5-3b-instruct-bf16.gguf`(6.18GB). 원본과 같은 bf16 값이다(양자화 없음) |
| AirLLM | 4.0.0을 의존성 없이 `.cache/airllm-pkgs`에(mlx·psutil만 더함, torch·transformers는 프로젝트 환경 것) |

## 측정 전에 알게 된 것 (개발 확인, 결과 아님)

- **llama.cpp의 GPU 사용**: bf16 3B를 Metal에 올리면 모든 층(`-ngl 99`), 자동 배치(기본), 24층, 12층 모두 `kIOGPUCommandBufferCallbackErrorOutOfMemory`로 실패했다. CPU만(`-ngl 0`) 돌았고, 16토큰 시험에서 약 14초/토큰, 최대 RSS 3.08GB, footprint 231MB였다(파일 매핑 페이지는 footprint에 들어가지 않는다).
  - 그래서 S9(속도)에서 llama.cpp 쪽 결과는 어느 정도 예상된다. 숨기지 않고 적는다.
- **AirLLM의 Mac 경로**: macOS에서 `AutoModel`은 무조건 MLX Llama 구현으로 간다. 이 구현은 어텐션에 편향이 없어 Qwen2(q·k·v 편향 있음)를 바르게 계산할 수 없다.
  - 그래서 AirLLM의 **일반 transformers 흘려 쓰기 클래스 `AirLLMBaseModel`을 `device="mps"`로 직접** 쓴다. AirLLM이 제공하는 공개 클래스이고, CUDA에서 기본으로 쓰이는 경로다. Mac에서 동작하는지는 모른다.
  - AirLLM은 처음 실행할 때 층별 파일을 디스크에 쓴다(약 모델 크기). 디스크 여유가 3.9GiB뿐이라, **llama.cpp 부분을 마친 뒤 GGUF를 지우고** AirLLM 부분을 돌린다. 쓴 층 파일은 측정 뒤 지운다.

## 설계

- **모델**: Qwen2.5-3B-Instruct bf16(5.75GiB). 8GB M1에서 다른 프로그램과 함께는 다 들어가지 않는 크기.
- **과제**: E033b 프롬프트 앞 두 개, 채팅 템플릿(HF 토크나이저가 만든 문자열을 llama.cpp에도 그대로 줌), 탐욕 생성 32토큰(끝 토큰 무시).
- **경우** (각각 새 프로세스, `/usr/bin/time -l`로 최대 RSS와 최대 footprint)

| 이름 | 방법 |
|---|---|
| MP | memopro `stream_model(device="mps", budget=1GiB)` + 일반 `generate` |
| MS | MP + 1.5B int4 초안(`draft_model`, `rtt.generate`) — memopro의 대표 경우 |
| L0 | llama.cpp bf16, CPU(`-ngl 0`), mmap 기본 |
| L12 | llama.cpp bf16, 12층 GPU |
| LA | llama.cpp bf16, 모든 층 GPU(`-ngl 99`) |
| A | AirLLM `AirLLMBaseModel(device="mps")`, 기본 설정(미리 읽기 켬) |

- **환경**: `MallocLargeCache=0`, 오프라인, `HF_HOME=.cache/hf`. MPS 워터마크는 라이브러리 기본값(0.01, 0162). 이전 실험과 달리 0.1을 넣지 않는다.
- **측정값**
  - s/token = 두 프롬프트의 (프롬프트 처리 + 32토큰 생성) 시간 합 ÷ 64. llama.cpp는 자기 출력의 prompt eval + eval 시간이다.
  - 불러오기 시간, 최대 RSS(파일 매핑 페이지 포함), 최대 footprint, 시스템 스왑 증감.
- **명령**
  1. `caffeinate -i .venv/bin/python -m experiments.e039_stream_infer.run --part llama`
  2. GGUF 삭제
  3. `caffeinate -i .venv/bin/python -m experiments.e039_stream_infer.run --part airllm`
- **결과**: `docs/research/data/e039/`(경우별 JSON, llama.cpp 원 출력, `summary.md`, `env_*.json`).

## 기준 (0179의 채점 조건으로 판정)

- 비교 대상마다 **완주한 설정 중 가장 빠른 것**을 그 대상의 대표로 쓴다(사용자가 고를 설정).
- 대상의 어떤 설정도 완주하지 못하면, 그 대상에 대해서는 memopro가 이긴 것으로 센다(E034에서 mlx-tune을 다룬 방식과 같다).

| # | 기준 |
|---|---|
| **S8** (메모리) | MS가 완주하고, MS의 최대 RSS ≤ 각 대상 대표의 최대 RSS |
| **S9** (속도) | MS의 s/token ≤ 각 대상 대표의 s/token |

- **보고만**: footprint, 스왑, 불러오기 시간, MP와의 출력 글자 일치(llama.cpp는 계산 커널이 달라 같지 않을 수 있다. 무손실 판정에는 쓰지 않는다).
- 판정은 `summary.md`의 규칙 그대로다. 실패해도 기준을 바꾸지 않는다.
- 스왑 오염 규칙(반복)은 쓰지 않는다. 판정이 스왑이 아니라 RSS와 속도이기 때문이다. 스왑은 보고한다.

## 논문 매핑

- **논문 P1 Evaluation (관련 시스템 비교)**: 같은 기기·같은 모델·같은 bf16 값에서 흘려 쓰는 추론 세 방식의 메모리와 속도.
