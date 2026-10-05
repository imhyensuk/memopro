# 0151. E034 결과: S3 충족 — 같은 16비트 LoRA에서 3B는 mlx-tune이 메모리 부족으로 못 돌리고 memopro는 1GiB 예산으로 20토큰/초, 1.5B는 mlx-tune이 2.6배 빠름

- **날짜**: 2026-10-05
- **유형**: experiment
- **상태**: 확정. 예측 Q1~Q4 모두 맞음. 성공 기준 S3 충족
- **관련 기록**: 0150 (사전 등록), 0149 (`memopro.finetune`), 0148 (S3)

## 실험 환경

| 항목 | 값 |
|---|---|
| 기계 | MacBook Air M1, 8GB, macOS 26.6.2. 다른 앱(Claude, Chrome, KakaoTalk)이 열린 평소 상태 |
| memopro | 커밋 `6bee8c3`, Python 3.14.0, torch 2.14.0, transformers 5.17.0, peft 0.21.0 |
| mlx-tune | 0.6.0(mlx 0.32.3, mlx-lm 0.32.0, transformers 5.18.0), 별도 venv `.cache/venv-mlx`(Python 3.12). 설치 목록 전체는 `env.json`의 `mlx_versions` |
| 실행 | 2026-10-05T01:48:20 시작, 약 2분(X3는 곧바로 실패) |

- 명령: `caffeinate -i .venv/bin/python -m experiments.e034_vs_mlx_tune.run --all`
- 원시 결과: `docs/research/data/e034/`. 들어 있는 것: `cases/*.json`, `summary.md`, `env.json`, `texts.json`, `mlx/<경우>/mlx_stdout.txt`, mlx-tune이 만든 데이터·어댑터, `supplement/`(아래 보충 실행).
- 실행기의 부모 프로세스는 `texts.json`을 만들 때 오프라인 환경 변수 없이 데이터셋을 불렀다. 그래서 HF Hub에 메타데이터 요청(인증 없음 경고)이 한 번 있었다. 데이터는 로컬 캐시에서 읽었고 내려받은 것은 없다.

## 결과 (`summary.md`)

| 경우 | 도구 | 모델 | 완주 | 정상 토큰/초 | 최대 footprint 증가 | 시스템 스왑 |
|---|---|---|---|---|---|---|
| X15 | mlx-tune | 1.5B | 10단계 | **118.9** | 3,632MiB | +227MiB |
| P15 | memopro (768MiB) | 1.5B | 10단계 | 45.4 | **1,411MiB** | −16MiB |
| X3 | mlx-tune | 3B | **실패: Metal 메모리 부족**(첫 단계 전) | — | — | +1,442MiB |
| P3 | memopro (1GiB) | 3B | **10단계** | **20.0** | **1,680MiB** | −376MiB |

| 예측 | 맞음 | 내용 |
|---|---|---|
| **Q1** 3B 메모리 | 예 | memopro +1,680MiB(≤ 2.5GiB). mlx-tune은 완주하지 못함 |
| **Q2** 3B 스왑 | 예 | mlx-tune 3B 실패 중 스왑 +1,442MiB |
| **Q3** 1.5B 속도 | 예 | mlx-tune 118.9 대 memopro 45.4토큰/초(**2.6배**) |
| **Q4** 3B 속도 | 예 | memopro 20.0토큰/초. mlx-tune은 완주하지 못함 |

- 단계별 손실은 두 도구의 학습률 일정(mlx-tune cosine, memopro 고정)과 조각 경계가 달라 비교하지 않는다(사전 등록대로). 값은 `summary.md`에 있다.

## X3 실패의 자세한 경위 (정직하게)

1. mlx-tune의 기본 경로(`_train_native` → `mlx_lm.tuner.train`)가 **첫 학습 단계 전, 시작 검증(`evaluate`)에서** 실패했다.
   - 오류: `RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)`.
   - mlx-tune은 기본으로 검증 집합을 학습 집합에서 복사해 만든다.
2. mlx-tune은 그다음 명령줄 학습(`mlx_lm.lora`)으로 넘어갔다. 그런데 실행기가 venv의 `bin`을 PATH에 넣지 않아 그 명령을 찾지 못했다(`FileNotFoundError`). **이것은 실행기의 결함**이다.
3. 그래서 **보충 실행**(사전 등록 밖)으로, mlx-tune이 실행하려던 명령을 PATH를 고쳐 그대로 돌렸다(`supplement/X3_cli.txt`).
   - 결과: 22초 만에 같은 Metal 메모리 부족으로 끝났다. 학습 단계는 시작하지 않았다.
   - `/usr/bin/time -l`로 잰 최대 footprint는 **6.46GB**, 시스템 스왑은 +482MiB였다.
   - 즉 실행기 결함과 무관하게, mlx-tune(과 그 아래 mlx-lm)은 이 기기에서 3B 16비트 LoRA를 시작하지 못한다.

## 해석

- **S3 충족**: 같은 기기·같은 과제(16비트 LoRA)로 Unsloth 호환 MLX 도구와 직접 비교했다.
- **memopro의 자리가 확인됐다.**
  - 모델이 기기에 들어가면(1.5B, 2.9GiB) mlx-tune이 2.6배 빠르다. memopro는 메모리를 39%만 쓴다(1,411 대 3,632MiB).
  - 들어가지 않으면(3B, 5.75GiB) mlx-tune은 시작하지 못한다(최대 6.46GB에서 Metal 메모리 부족). memopro는 1GiB 예산으로 20토큰/초로 완주한다.
  - 이것이 0148 목표 문장의 "같은 기기에서 남들이 못 돌리는 크기"의 첫 직접 근거다.
- **속도 격차의 원인(추정)**: memopro는 가중치를 단계마다 다시 읽는다(재진입 체크포인팅 때문에 약 2회). PyTorch MPS bf16 계산도 MLX보다 느릴 수 있다(측정 안 함). 1.5B를 768MiB가 아니라 더 큰 예산으로 돌리면 격차가 줄 수 있다(측정 안 함).
- 3B의 20.0토큰/초는 E028c(LoRA q·v, 11.9토큰/초)보다 빠르다. 하지만 텍스트·대상 층·손실 조각이 달라 같은 조건의 비교가 아니다.

## 한계

- 한 기기, 10단계, 한 번 측정. 다른 앱이 열린 평소 상태.
- mlx-tune의 기본값(검증 집합 복사, cosine 일정)을 그대로 썼다.
  - 검증을 끄면 시작 검증의 메모리 부족은 피할 수 있었을지 모른다. 하지만 6.46GB 최대 footprint(8GB 기기)로 보아 학습 단계도 넘쳤을 가능성이 크다(측정 안 함).
- Unsloth 본체(CUDA)와의 비교는 아니다.

## 논문 매핑

- **Evaluation (G4 경쟁)**: 표 두 개. "들어가면 MLX가 빠르고, 안 들어가면 memopro만 돈다"라는 경계.
- **Threats**: 위 한계. 실행기 결함(PATH)과 보충 실행으로 확인한 경위.
