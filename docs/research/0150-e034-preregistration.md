# 0150. E034 사전 등록: 성공 기준 S3 — 같은 16비트 LoRA 과제를 mlx-tune과 memopro로 (8GB M1)

- **날짜**: 2026-10-05
- **유형**: experiment (사전 등록)
- **상태**: 확정. 실행 전
- **관련 기록**: 0148 (S3), 0149 (`memopro.finetune`), 0125 (경쟁자 조사: Unsloth Mac = MLX, mlx-tune), 0138 (G4-B1)

## 사용자 지시와 설치

- "순서대로 전부 진행해." 0148 뒤 제안의 4번이 "경쟁자 비교(설치 허락 필요)"였으므로, 이 지시를 설치 허락으로 받았다.
- 프로젝트 환경과 섞이지 않도록 따로 만든 가상환경 `.cache/venv-mlx`(Python 3.12, git 추적 안 함, 965MB)에 `mlx-tune` 0.6.0을 설치했다. 함께 깔린 것: mlx 0.32.3, mlx-lm 0.32.0, transformers 5.18.0.
- **mlx-tune**: "Unsloth를 대체하는 MLX 구현"이다(패키지 설명). Unsloth 호환 API(`FastLanguageModel`, `SFTTrainer`)를 쓰고, 학습은 mlx-lm 위에서 한다.
- **Unsloth 자체를 고르지 않은 이유**: Unsloth는 CUDA/Triton 커널 중심이고, Mac 지원은 MLX 경로다(0125). 같은 기기(M1)의 기준선으로는 mlx-tune이 맞다.
- **양자화 경우는 뺐다**: mlx-tune의 `load_in_4bit`은 미리 양자화된 모델(mlx-community, 추가 다운로드)을 요구한다(코드 확인). 그리고 S3는 같은 과제(16비트 LoRA)의 비교다.

## 개발 확인 (결과 아님)

- mlx-tune 1.5B를 3단계 돌렸다(스크래치). 완주했다.
- 단계 2·3이 101·120토큰/초, 최대 footprint 3.85GB였다. 로그 형식(색 있는 표)을 확인해 파서를 맞췄다.
- 3B는 사전 등록 전에 돌리지 않았다.

## 설계

- **기기**: M1 8GB, macOS 26.6.2. 다른 앱(Claude, Chrome, KakaoTalk)이 열린 평소 상태다. 두 도구에 같은 조건이다.
- **과제(두 도구 같음)**
  - Qwen2.5-1.5B·3B-Instruct, bf16 원본(로컬 캐시).
  - LoRA r 8, alpha 16, 대상 q·k·v·o_proj, 학습률 2e-4, 배치 1.
  - 10단계. 각 단계는 WikiText-2 앞부분 128토큰 글 하나 + 끝 토큰 = 129토큰이다. 글 10개는 `docs/research/data/e034/texts.json`에 있다.
- **도구별 설정**
  - mlx-tune: 문서의 기본 경로(`FastLanguageModel.from_pretrained` → `get_peft_model` → `SFTTrainer(SFTConfig(..., grad_checkpoint=True))`).
    - 학습률 일정(cosine), 검증 계산, 어댑터 저장은 mlx-tune의 기본값 그대로다.
    - 속도는 mlx-lm이 보고하는 단계별 토큰/초다.
  - memopro: `memopro.finetune(..., budget=768MiB(1.5B) / 1GiB(3B), seq_len=129)`(MPS). 속도는 단계별 129 / 단계 시간이다.
- **경우**(실행 순서, 경우마다 새 프로세스, 시간 한도 45분)

| 경우 | 도구 | 모델 |
|---|---|---|
| X15 | mlx-tune | 1.5B |
| P15 | memopro | 1.5B |
| X3 | mlx-tune | 3B |
| P3 | memopro | 3B |

- **지표**
  - 완주 여부.
  - 정상 토큰/초(단계 2~10 평균).
  - 최대 footprint 증가(`proc_pid_rusage` V4 생애 최대값, GPU 포함).
  - 시스템 스왑 증가.
  - 단계별 손실. 학습률 일정이 달라 비교하지 않고 보고만 한다.
- 오염 규칙(반복)은 쓰지 않는다. 스왑은 도구의 메모리 사용이 낳는 결과의 일부라서 그대로 기록한다.
- **명령**: `caffeinate -i .venv/bin/python -m experiments.e034_vs_mlx_tune.run --all`
- **결과 위치**: `docs/research/data/e034/`(`cases/*.json`, `mlx/*/mlx_stdout.txt`, `summary.md`, `env.json`, 설치 목록 포함)

## 예측 (사전 등록, 판정 코드 `summarize`)

| # | 예측 |
|---|---|
| **Q1** 3B 메모리 | memopro footprint 증가 ≤ 2.5GiB, 그리고 mlx-tune은 ≥ 5.5GiB이거나 완주하지 못한다 |
| **Q2** 3B 스왑 | mlx-tune 3B가 스왑을 1GiB 넘게 늘리거나 완주하지 못한다 |
| **Q3** 1.5B 속도 | 기기에 들어가는 1.5B에서는 mlx-tune이 memopro의 2배 이상 빠르다 |
| **Q4** 3B 속도 | 3B에서는 memopro가 mlx-tune 이상이다(또는 mlx-tune이 완주하지 못한다) |

- **S3의 완료 조건**: 네 경우를 실행하고 결과를 기록하면 S3("같은 과제로 직접 비교")를 채운다. 예측이 틀려도 결과는 그대로 보고한다. 예측은 memopro의 우위 주장이 아니라 검증할 가설이다.

## 논문 매핑

- **Evaluation (G4 경쟁)**: 같은 기기·같은 과제에서 MLX 기반 Unsloth 대체(mlx-tune)와 memopro의 메모리·속도. 들어가는 크기(1.5B)와 넘는 크기(3B).
- **Threats**: 한 기기, 10단계, 다른 앱이 열린 평소 상태, 도구별 기본 설정 차이(학습률 일정, 검증).
