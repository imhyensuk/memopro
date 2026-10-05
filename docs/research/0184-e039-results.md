# 0184. E039 결과: S8·S9 통과 — 8GB M1의 3B bf16 추론에서 memopro 1.09초/토큰·RSS 2.1GB, llama.cpp는 CPU만 돌아 14.75초/토큰·3.3GB, AirLLM은 완주 못 함

- **날짜**: 2026-10-05
- **유형**: experiment
- **상태**: 확정. 채점표 S8·S9 ✔
- **관련 기록**: 0180 (사전 등록), 0179 (채점표), 0152 (E033c)

## 실험 환경

| 항목 | 값 |
|---|---|
| 기계 | MacBook Air M1, 8GB, macOS 26.6.2, 전원 연결 |
| 코드 | 커밋 `b3da8b6`(llama 부분), AirLLM 부분도 같은 실행기 |
| 실행 | llama 부분 2026-10-05T17:56 시작 약 40분, AirLLM 둘째 시도 18:19 |
| 원시 결과 | `docs/research/data/e039/` (경우별 JSON, llama.cpp 원 출력 `raw/`, `summary.md`, `env_*.json`) |

- 명령: 0180 그대로(`--part llama`, GGUF 삭제, `--part airllm`).

## 판정 (`summary.md`)

| 경우 | 완주 | s/token | 최대 RSS | 최대 footprint | 스왑 | 불러오기 |
|---|---|---|---|---|---|---|
| **MS** memopro + int4 초안 | 예 | **1.09** | **2,138MiB** | 2,941MiB | +592MiB | 10.3초 |
| MP memopro 일반 | 예 | 3.42 | 1,521MiB | 1,627MiB | −745MiB | 3.1초 |
| L0 llama.cpp CPU | 예 | 14.75 | 3,309MiB | 189MiB | −832MiB | 25.2초 |
| L12 llama.cpp 12층 GPU | 아니오 | — | — | — | — | Metal 메모리 부족 |
| LA llama.cpp 모든 층 GPU | 아니오 | — | — | — | — | Metal 메모리 부족 |
| A AirLLM `AirLLMBaseModel(mps)` | 아니오 | — | — | — | — | AirLLM 내부 오류 |

- **S8 (메모리): 통과.** MS의 최대 RSS 2,138MiB ≤ llama.cpp 대표(L0) 3,309MiB. AirLLM은 완주하지 못했다.
- **S9 (속도): 통과.** MS 1.09초/토큰 ≤ L0 14.75초/토큰(13.5배). 초안 없는 MP(3.42초)도 L0보다 4.3배 빠르다.
- MS의 출력은 MP와 2/2 같았다(무손실 확인).

## 실행 기록 (기준은 그대로)

- **AirLLM 첫 시도는 측정 전 준비 오류**였다. AirLLM이 선언한 의존성 `sentencepiece`를 격리 폴더에 넣지 않아 `import`에서 멈췄다(연구자 실수). 기록은 `run1_airllm_setup_error/`에 두고, 의존성을 넣어 같은 명령으로 다시 돌렸다.
- **둘째 시도에서 AirLLM은 층 파일 5.7GB를 디스크에 쓴 뒤**, 자기 코드 안에서 멈췄다.
  - 오류: `airllm_base.py`의 `_should_load_verbatim`, `AttributeError: 'dict' object has no attribute 'is_floating_point'`(임베딩 상태를 옮기는 단계).
  - Qwen2.5-3B의 묶인 임베딩(tie_word_embeddings)과 관련된 것으로 보이지만 확인하지 않았다.
  - 사전 등록의 규칙대로 "완주 못 함"으로 센다. 다만 이것은 **AirLLM 4.0.0의 일반 경로가 이 모델·MPS에서 동작하지 않았다**는 뜻이지, AirLLM 방식의 한계를 잰 것은 아니다.
  - 층 파일은 측정 뒤 지웠다.
- llama.cpp의 GPU 설정은 개발 확인(0180)과 같이 모두 Metal 메모리 부족이었다.

## 해석

- **같은 bf16 값, 같은 기기에서 memopro가 메모리와 속도 모두 앞섰다.**
  - llama.cpp는 모델 파일을 mmap해 OS 페이징에 맡긴다. 8GB 기기에서 6.2GB 모델은 쪽이 계속 밀려나고 다시 읽혀 CPU에서 14.75초/토큰이었다.
  - memopro는 예산 1GiB 안에서 GPU로 계산하고, 초안으로 가중치 읽기 횟수를 줄였다.
- **footprint와 RSS의 차이**: llama.cpp의 footprint(189MiB)가 작은 것은 파일 매핑 쪽이 footprint에 들어가지 않기 때문이다. 실제로 차지한 물리 메모리는 RSS(3.3GB)다. 그래서 사전 등록에서 RSS를 기준으로 정했다. memopro의 footprint(2.9GB)는 GPU 할당과 초안을 포함한다.
- **공정성의 한계**
  - llama.cpp에도 추측 디코딩(`llama-speculative`, 초안 모델)이 있다. 이번에는 시험하지 않았다. 다만 초안 없는 MP도 L0보다 4.3배 빠르다.
  - 메모리가 넉넉한 기기라면 llama.cpp(GPU)가 훨씬 빠를 것이다. 이 비교는 "모델이 기기 메모리에 다 들어가지 않는 경우"에 한정된다.
  - 시스템 스왑은 MS에서 +592MiB였다(기계 상태, 판정 기준 아님).
- **글 비교(보고만)**: llama.cpp의 글을 뽑는 실행기 코드가 특수 토큰이 빠진 출력과 맞지 않아 비어 있다(실행기 결함, 판정과 무관). 원 출력을 보면 첫 프롬프트는 15번째 단어 근처에서 갈린다(memopro "data", llama.cpp "elements"). 계산 커널이 달라 생기는 차이로 보며, 무손실 판정에는 쓰지 않는다.

## 채점표 변화 (0179)

- S8: ✘ → ✔, S9: ✘ → ✔. 부문 III 5/19 → 7/19.

## 논문 매핑

- **논문 P1 Evaluation (관련 시스템 비교)**: 같은 bf16 3B를 8GB M1에서 — memopro 1.09초/토큰·2.1GB 대 llama.cpp 14.75초/토큰·3.3GB, AirLLM 동작 안 함.
- **Threats**: llama.cpp 추측 디코딩 미시험, AirLLM은 Mac 일반 경로의 결함으로 판정 불가에 가까움, 한 모델·한 기기.
