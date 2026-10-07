"""Assemble one Colab notebook per part (0092): examples/colab_t4_<part>.ipynb,
each a short guide and one self-contained code cell.

The code cell = its settings + common.py + the worker sources (written to disk at run time) +
its body. Usage: .venv/bin/python examples/colab_t4/build.py [--out-dir DIR for .py copies]
[--nb-dir DIR for the notebooks, default examples/]
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKER_FILES = ["worker_common.py", "worker_train.py", "worker_infer.py", "worker_multi.py",
                "app_naive_load.py"]
# more files a notebook writes next to the workers: name on Colab -> path in this repository
EXTRA_WORKERS = {
    "suite": {
        "worker_suite.py": HERE / "worker_suite.py",
        "worker_data.py": HERE / "worker_data.py",
        "image.py": ROOT / "experiments/e027_transparent/workload.py",
        "common.py": ROOT / "experiments/e040_workloads/common.py",
        "dataframe.py": ROOT / "experiments/e040_workloads/dataframe.py",
        "classify.py": ROOT / "experiments/e040_workloads/classify.py",
        "simulate.py": ROOT / "experiments/e040_workloads/simulate.py",
        "preload_smoke.c": ROOT / "crates/memopro-preload/tests/smoke.c",
    },
    "unsloth": {
        "worker_suite.py": HERE / "worker_suite.py",
        "worker_unsloth.py": HERE / "worker_unsloth.py",
    },
    "full": {
        "worker_suite.py": HERE / "worker_suite.py",
        "worker_unsloth.py": HERE / "worker_unsloth.py",
        "worker_data.py": HERE / "worker_data.py",
        "worker_memopro_run.py": HERE / "worker_memopro_run.py",
        "worker_pytest.py": HERE / "worker_pytest.py",
        "image.py": ROOT / "experiments/e027_transparent/workload.py",
        "passes.py": ROOT / "experiments/e047_enable/passes.py",
        "common.py": ROOT / "experiments/e040_workloads/common.py",
        "dataframe.py": ROOT / "experiments/e040_workloads/dataframe.py",
        "classify.py": ROOT / "experiments/e040_workloads/classify.py",
        "simulate.py": ROOT / "experiments/e040_workloads/simulate.py",
    },
}

COMMON_PREP = """
**준비 (처음 한 번, 세 노트북 공통)**
1. 런타임 → 런타임 유형 변경 → **T4 GPU**.
2. Google Drive에 `memopro_colab/install/` 폴더를 만들고 memopro 소스 배포본 `memopro-0.1.0a1.tar.gz`를 올린다(Linux wheel이 있으면 그것을 쓴다). 소스 배포본은 첫 실행 때 Rust로 빌드한다(3~5분).
   - 또는 Colab 왼쪽 열쇠(보안 비밀)에 `GITHUB_TOKEN`을 **직접** 등록하면 저장소에서 설치한다.
3. 셀을 실행한다. Drive 연결 허용 창이 뜨면 허용한다. 설정은 셀 맨 위(`SETTINGS`)에서 바꾼다.

**결과**: `MyDrive/memopro_colab/results/{cell}/<실행 id>/`
- `summary.md`·`summary.csv`: 비교 표와 읽는 법
- `cases/*.json`: 경우마다 모든 측정값(메모리, 시간, memopro 보고, 오류와 추적)
- `logs/*.log`: 경우마다 프로세스 출력 전체
- `timelines/*.csv`: 0.5초 간격 GPU·호스트 메모리와 GPU 사용률
- `env.json`: GPU, 판, memopro 빌드 확인
- 같은 이름의 `.zip`: 폴더 전체(공유용)

**끊겼을 때**: 같은 셀을 다시 실행하면 끝난 경우는 건너뛰고 이어서 한다(`NEW_RUN = True`면 새로 시작). 모델은 Drive의 `hf_cache/`에 남아 다시 받지 않는다. 세 노트북은 같은 Drive 폴더와 모델 캐시를 함께 쓴다.

**공정성**: 모든 경우는 **새 프로세스**에서 돈다(앞 경우의 메모리가 섞이지 않게, 0055). 여러 모델을 **동시에** 한 GPU에서 재면 서로의 측정을 오염시키므로(0085), 여러 모델은 목록으로 차례로 잰다.
"""

NOTEBOOKS = {
    "train": ("colab_t4_train.ipynb", "cell_train.py", """# memopro Colab T4 재검증 ① 학습·개발 과정 (docs/research/0091, 0092)

GPT-2, Qwen2.5-0.5B·1.5B 전체 미세조정(fp32 AdamW, WikiText-2, 같은 시드·데이터·단계, dropout 끔)을 다섯 방식으로 비교한다: 일반 PyTorch(기준), fp16 AMP, HF gradient checkpointing, accelerate 배치 탐색, memopro `train_session`(+ lossless). 전체 T4와 메모리 상한(작은 GPU 흉내) 두 조건, `memopro.check` 예측 대 실측, 7B QLoRA(표준 방식 대 memopro).

- 예상 시간: 2~3시간(7B 첫 다운로드 약 15GB 포함)
- 판정(0091): T1 같은 모델 학습(손실 ≤ 1e-3, 파라미터 ≤ 1e-5), T2 작은 GPU에서 완주, T3 `check` ±15%, T4 안 될 때 제안과 함께 거절
"""),
    "infer": ("colab_t4_infer.ipynb", "cell_infer.py", """# memopro Colab T4 재검증 ② 로컬 AI 모델 구동 (docs/research/0091, 0092)

Qwen2.5 1.5B·3B·7B·14B Instruct를 흔한 방식(fp16 `device_map="auto"`, bitsandbytes 8비트·4비트)과 `memopro.load`(기본·`quality="low"`)로 불러와 비교한다: 불러오기 시간, 가중치 위치, GPU·호스트 최대 사용, 512·2048토큰 프롬프트 첫 토큰 시간, 디코딩 속도, WikiText-2 perplexity, 채팅 답변 일치(가장 정확한 방법 대비).

- 예상 시간: 2~3시간(첫 실행은 모델 다운로드 약 55GB 포함)
- 판정(0091): I1 memopro는 OOM으로 끝나지 않음(불러오거나 제안과 함께 거절), I2 품질·속도는 그대로 보고
"""),
    "multi": ("colab_t4_multi.ipynb", "cell_multi.py", """# memopro Colab T4 재검증 ③ 여러 모델·프로세스 수준 기능 (docs/research/0091, 0092)

doctor 정확도, 모델 3개(fp16 합 약 17GB > T4)를 한 GPU에서 번갈아 쓰기(모두 올려 두기 / 지우고 다시 불러오기 / memopro 동면), 수정하지 않은 스크립트를 `python` 대 `memopro run`, 최근 변경의 CUDA·Linux 회귀 검사 7종.

- 예상 시간: 약 1시간
- 판정(0091): R1 회귀 7종 통과(배포 전 필수), M1 동면 뒤 같은 답, M2·M3 보고
"""),
    "remeasure": ("colab_t4_remeasure.ipynb", "cell_remeasure.py", """# memopro Colab T4 재측정: 0094 수정(D1~D4) 확인 (docs/research/0096, E022)

Colab 7차(0093)에서 드러난 결함이 고쳐졌는지 **통합 셀 하나**로 확인한다. 7차에서 받은 모델이 Drive `hf_cache/`에 있으면 다시 받지 않는다(약 1~1.5시간).

| 확인 | 경우 | 기준(0096) |
|---|---|---|
| R1 | 회귀 검사 7종 | 모두 통과(γ는 `run`에서 기본 꺼짐) |
| D1 | 7B QLoRA 표준 방식 대 memopro | `train_session` 계획이 건너뛰어지지 않고 완주 |
| D2 | 1.5B·3B HF fp16 대 memopro | memopro가 fp16을 고르고, 2048토큰 첫 토큰 시간 ≤ HF fp16의 1.2배 |
| D3 | 7B HF bnb4·bnb8 대 memopro(기본·low) | low는 int4, 디코딩 ≥ HF bnb4의 0.9배. 기본은 int8 유지 |
| D4 | 수정하지 않은 스크립트를 `memopro run`으로(1.5B·7B) | 7B는 정책 적용(γ 기록 없음), 1.5B는 개입 안 함 |

**반드시 새 sdist를 쓴다**: Drive `memopro_colab/install/`의 `memopro-0.1.0a1.tar.gz`를 이번에 받은 파일로 **바꿔 놓는다**(0094 수정 포함). 셀 첫 출력의 `build_has`에서 `0094_fixes: true`를 확인한다. 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한 뒤 실행한다.

결과 요약(`summary.md`) 맨 위에 판정 표(pass/fail)가 나온다.
"""),
    "remeasure2": ("colab_t4_remeasure2.ipynb", "cell_remeasure2.py", """# memopro Colab T4 재측정 2: 0099·0100 수정 확인 (docs/research/0101, E023)

`train_session`이 이제 메모리가 허락하는 만큼 큰 micro-batch를, 고르게, OOM 없이 고르는지와 `load`·`memopro run`이 int4의 속도·품질 교환을 알려 주는지 **통합 셀 하나**로 확인한다. 앞선 실행에서 받은 모델이 Drive `hf_cache/`에 있으면 다시 받지 않고, 7B는 로컬 디스크에 한 번만 복사한다(약 45~60분).

| 확인 | 경우 | 기준(0101) |
|---|---|---|
| R1·D9 | 회귀 검사 7종 | 모두 통과, CUDA int4 품질 문구는 bitsandbytes 수치 |
| T1 | GPT-2 배치 16, 전체 T4 | 고른 분할(7차: 15+1), 재시도 0, HF checkpointing과 손실 차이 ≤ 1e-5 |
| T2 | GPT-2, 30% 상한 | OOM 재시도 없이 완주(7차: 3 → OOM → 2), 손실 차이 ≤ 1e-5 |
| T3 | Qwen2.5-0.5B 배치 8, 전체 T4 | micro ≥ 2(7차: 1), 재시도 0, 고정 micro 1과 손실 차이 ≤ 1e-5 |
| T4 | 7B QLoRA 배치 4 | micro ≥ 2(E022: 1), 재시도 0, 속도 ≥ 표준의 0.95배, 손실 차이 ≤ 5e-3 |
| O1 | 7B 기본 품질 `load`, `memopro run` | int8 유지, 보고에 `quality='low'`(run은 `--quality low`) → int4 안내 |

**반드시 새 sdist를 쓴다**: Drive `memopro_colab/install/`의 `memopro-0.1.0a1.tar.gz`를 이번에 받은 파일로 **바꿔 놓는다**. 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한 뒤 실행한다. 빌드에 0099·0100 수정이 없으면 셀이 측정 전에 멈추고 이유를 알려 준다(`build_has`의 `0099_fixes`·`0100_fixes`).

결과 요약(`summary.md`) 맨 위에 판정 표(pass/fail)가 나온다.
"""),
    "qlora": ("colab_t4_qlora.ipynb", "cell_qlora.py", """# memopro Colab T4 재측정 3: 7B QLoRA만 (docs/research/0104, E024)

0103(여유 계수 1.3, 학습 시작 때 CUDA 캐시를 비우고 실측)이 7B QLoRA에서 어떻게 작동하는지 **통합 셀 하나**로 확인한다. 표준 방식(bnb nf4 + 이중 양자화 + peft)과 memopro(`load(quality="low")` + peft + `train_session`)를 같은 데이터·단계로 비교한다. 7B가 Drive `hf_cache/`에 있으면 다시 받지 않는다(약 20~25분).

| 확인 | 기준(0104) |
|---|---|
| Q1 | memopro가 OOM 재시도 없이 20단계 완주 |
| Q2 | memopro의 최대 예약 메모리 ≤ memopro가 잡은 예산(E023: 13.90 > 13.75GiB) |
| Q3 | 속도 ≥ 표준 방식의 0.9배, 손실 차이 ≤ 5e-3 |

**반드시 새 sdist를 쓴다**: Drive `memopro_colab/install/`의 `memopro-0.1.0a1.tar.gz`를 이번에 받은 파일로 **바꿔 놓는다**. 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한 뒤 실행한다. 빌드에 0103이 없으면 셀이 측정 전에 멈춘다.

결과 요약(`summary.md`)에 판정 표와, 학습 시작 전후의 캐시 상태(memopro가 돌려준 양)가 나온다.
"""),
    "suite": ("colab_t4_suite.ipynb", "cell_suite.py", """# memopro Colab T4 통합 시험: 언어 모델·비전·데이터프레임·시뮬레이션 (docs/research/0197, E042)

셀 **하나**로 아래를 차례로 잰다. 모든 경우는 새 프로세스에서 돌고, 결과와 모델은 Google Drive에 남는다(끊겨도 이어서 한다).

| 갈래 | 경우 | 무엇을 보나 |
|---|---|---|
| 언어 모델 | Qwen2.5-3B·7B 생성, 16비트 LoRA | 그냥 GPU에 올리기 대 memopro(호스트 메모리에서 GPU로 흘려 쓰기). 7B는 T4(15GB)에 그냥은 안 들어간다 |
| 비전 | ResNet-152(CNN), DINOv2-giant(11억 파라미터) 추론, DINOv2 LoRA | 예산 약 1/4·1/2에서 결과가 그냥 실행과 같은가 |
| 데이터 | 영상 묶음, pandas, scikit-learn, 열 확산 시뮬레이션 | 코드를 고치지 않고 `memopro-preload`로 메모리 절반에서 같은 결과 |

**준비 (이 노트북만 다름)**
1. 런타임 → 런타임 유형 변경 → **T4 GPU**.
2. Google Drive `memopro_colab/install/`에 **소스 묶음 `memopro-src.tar.gz`**를 올린다(memopro와 `memopro-preload`를 함께 빌드한다, 첫 실행 5~8분). 예전 sdist는 CUDA 흘려 쓰기와 preload가 없어 쓰지 않는다. 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한다.
3. Drive에 모델용 공간이 약 **26GB** 필요하다(3B 6GB, 7B 15GB, DINOv2 4GB, ResNet 0.2GB). 모자라면 설정의 `MODEL_CACHE = "local"`로 바꾼다(모델을 이 세션의 로컬 디스크에만 받는다).
4. 셀을 실행한다. 예상 시간 2~3시간(첫 다운로드 포함).

결과 요약(`summary.md`) 맨 위에 판정 표(pass/fail)가 나온다. 판정 기준은 docs/research/0197에 미리 정해 두었다.
"""),
    "unsloth": ("colab_t4_unsloth.ipynb", "cell_unsloth.py", """# memopro 대 Unsloth: 같은 16비트 LoRA (docs/research/0227, E046)

같은 T4에서 같은 작업(WikiText-2, 512토큰, 배치 1, 10단계, LoRA r 8 q/k/v/o, AdamW 2e-4)을 세 방식으로 잰다: Hugging Face PEFT(그냥), memopro(가중치 흘려 쓰기), Unsloth(별도 가상환경). 모델은 Qwen2.5-3B와 7B. 7B는 16비트로 T4에 들어가지 않는다.

| 확인 | 기준(0227) |
|---|---|
| U1 (관문) | 7B 16비트: Unsloth는 완주 못 하고 memopro는 두 예산 모두 완주 |
| U2 (관문) | memopro 손실이 예산과 무관하게 비트 동일 |
| U3·U4·U5 | 3B 속도·메모리, 7B Unsloth 4비트 대 memopro 16비트, 손실 차이(보고) |

- 예상 시간: 45~60분(Unsloth 설치 5~10분 포함)
- 준비: 런타임 **T4 GPU**, Drive `memopro_colab/install/`에 소스 묶음 `memopro-src.tar.gz`(이미 있으면 그대로, 빌드는 Drive에 보관된 것을 다시 쓴다)
"""),
    "full": ("colab_t4_full.ipynb", "cell_full.py", """# memopro 통합 시험: 현재 코드 전체 (docs/research/0237, E048)

현재 코드(`memopro.enable`, 프로세스 상한, 예측, 흘려 쓰기, Unsloth 비교)를 Colab T4 **셀 하나**로 잰다. **2시간 30분 안에 끝나도록** 셀 전체 예산을 135분으로 두고, 시간이 모자라면 뒤쪽 경우를 `skipped (time budget)`로 남기고 넘어간다(다시 실행하면 이어서 한다).

| 순서 | 갈래 | 무엇을 보나 | 예상 시간 |
|---|---|---|---|
| ① | 단위 시험 | enable·투명 페이징·adopt·흘려 쓰기 시험을 T4에서(CUDA 포함) | 5분 |
| ② | enable (Linux) | E047을 Linux에서: 영상 묶음·4패스 배열을 넉넉함/0.75/0.5 상한으로. 결과 동일, 상한, 오버헤드, 예측 | 5분 |
| ③ | Unsloth 비교 | 같은 16비트 LoRA를 PEFT·memopro·Unsloth로(Qwen2.5-3B·7B) | 50~60분 |
| ④ | DINOv2 | 비전 모델 흘려 쓰기 추론·LoRA | 10분 |
| ⑤ | 데이터 | E027·E040 프로그램을 `memopro run --budget 절반`으로(경우당 최대 10분) | 25분 |

- 준비: 런타임 **T4 GPU**. Drive `memopro_colab/install/`에 이번 소스 묶음 `memopro-src.tar.gz`를 **덮어쓰기**(현재 코드, 첫 실행 때 빌드 5~8분, 이후 Drive에 보관된 빌드 재사용). 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**.
- 결과: `MyDrive/memopro_colab/results/full/<실행 id>/summary.md`(판정 표), 같은 이름의 `.zip`.
"""),
}


SUITE_PARTS = {  # split notebooks of the suite (one part each, same run folder)
    'suite_lm': ('["lm"]', """# memopro Colab T4 통합 시험 ① 언어 모델 (docs/research/0197, E042)

Qwen2.5-3B·7B의 생성과 16비트 LoRA를 그냥 GPU에 올리기 대 memopro(호스트 메모리에서 GPU로 흘려 쓰기)로 잰다. 7B는 T4(15GB)에 그냥은 안 들어간다.

- 예상 시간: 약 40~70분(첫 실행은 모델 다운로드 약 21GB 포함)
- 판정: G(생성 동일), L(LoRA 손실이 예산과 무관하게 비트 동일)

**나눈 노트북 네 개**(①언어 모델 ②비전 ③속도 ④데이터)는 Drive의 **같은 결과 폴더**(`results/suite/`)를 쓴다. 아무 순서로 하나씩 실행하면 되고, 끝난 경우는 건너뛴다. 동시에 두 세션에서 돌리지 않는다. `summary.md`는 지금까지 끝난 부분을 모두 판정하고, 아직 안 돌린 부분은 `n/a (not run yet)`로 나온다.

**준비**: 런타임 → 런타임 유형 변경 → **T4 GPU**. Drive `memopro_colab/install/`에 소스 묶음 `memopro-src.tar.gz`(첫 실행 때 빌드 5~8분). 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한다.
"""),
    'suite_vision': ('["vision"]', """# memopro Colab T4 통합 시험 ② 비전 (docs/research/0197, E042)

ResNet-152 추론, DINOv2-giant 추론과 LoRA를 예산 약 1/4·1/2에서 잰다.

- 예상 시간: 약 20~40분(DINOv2 다운로드 약 4GB 포함)
- 판정: V(추론 출력 동일), VL(LoRA 손실이 예산과 무관하게 비트 동일)

**나눈 노트북 네 개**(①언어 모델 ②비전 ③속도 ④데이터)는 Drive의 **같은 결과 폴더**(`results/suite/`)를 쓴다. 아무 순서로 하나씩 실행하면 되고, 끝난 경우는 건너뛴다. 동시에 두 세션에서 돌리지 않는다. `summary.md`는 지금까지 끝난 부분을 모두 판정하고, 아직 안 돌린 부분은 `n/a (not run yet)`로 나온다.

**준비**: 런타임 → 런타임 유형 변경 → **T4 GPU**. Drive `memopro_colab/install/`에 소스 묶음 `memopro-src.tar.gz`(첫 실행 때 빌드 5~8분). 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한다.
"""),
    'suite_speed': ('["speed"]', """# memopro Colab T4 통합 시험 ③ 속도(GPU 캐시) (docs/research/0197, E042)

GPU 캐시와 미리 복사(0201)를 켠 memopro로 3B·7B 생성과 DINOv2 추론 속도를 잰다. 같은 출력인지도 본다.

- 예상 시간: 약 20~40분
- 판정: S(출력 동일, 3B·DINOv2는 그냥의 2배 이내, 7B는 2.0초/토큰 이내, 0202)

**나눈 노트북 네 개**(①언어 모델 ②비전 ③속도 ④데이터)는 Drive의 **같은 결과 폴더**(`results/suite/`)를 쓴다. 아무 순서로 하나씩 실행하면 되고, 끝난 경우는 건너뛴다. 동시에 두 세션에서 돌리지 않는다. `summary.md`는 지금까지 끝난 부분을 모두 판정하고, 아직 안 돌린 부분은 `n/a (not run yet)`로 나온다.

**준비**: 런타임 → 런타임 유형 변경 → **T4 GPU**. Drive `memopro_colab/install/`에 소스 묶음 `memopro-src.tar.gz`(첫 실행 때 빌드 5~8분). 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한다.
"""),
    'suite_data': ('["data"]', """# memopro Colab T4 통합 시험 ④ 데이터 프로그램 (docs/research/0197, E042)

영상 묶음, pandas, scikit-learn, 열 확산 시뮬레이션을 코드 수정 없이 그냥 실행 대 `memopro-preload`로 메모리 절반에서 실행한다(CPU만 씀).

- 예상 시간: 약 30~60분
- 판정: D(같은 결과, 프로세스 최대 메모리 ≤ 절반 + 64MiB, overruns 0)

**나눈 노트북 네 개**(①언어 모델 ②비전 ③속도 ④데이터)는 Drive의 **같은 결과 폴더**(`results/suite/`)를 쓴다. 아무 순서로 하나씩 실행하면 되고, 끝난 경우는 건너뛴다. 동시에 두 세션에서 돌리지 않는다. `summary.md`는 지금까지 끝난 부분을 모두 판정하고, 아직 안 돌린 부분은 `n/a (not run yet)`로 나온다.

**준비**: 런타임 → 런타임 유형 변경 → **T4 GPU**. Drive `memopro_colab/install/`에 소스 묶음 `memopro-src.tar.gz`(첫 실행 때 빌드 5~8분). 이전 세션에서 memopro를 설치했다면 **런타임을 다시 시작**한다.
"""),
}


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip() or "unknown"


def workers_block(extra=None):
    lines = ["WORKERS = {"]
    files = {name: HERE / name for name in WORKER_FILES} | (extra or {})
    for name, path in files.items():
        src = path.read_text(encoding="utf-8")
        assert "'''" not in src, name
        lines.append(f"    {name!r}: r'''{src}''',")
    lines.append("}")
    return "\n".join(lines)


def compose(cell_file, extra=None):
    text = (HERE / cell_file).read_text(encoding="utf-8").replace("@@COMMIT@@", commit())
    head, body = text.split("# @@COMMON@@\n")
    common = (HERE / "common.py").read_text(encoding="utf-8")
    return head + common + "\n\n" + workers_block(extra) + "\n\n" + body


def notebook(title_md, src):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": title_md.splitlines(True)},
             {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": src.splitlines(True)}]
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]) if "--out-dir" in sys.argv else None
    nb_dir = Path(sys.argv[sys.argv.index("--nb-dir") + 1]) if "--nb-dir" in sys.argv else ROOT / "examples"
    jobs = [(name, file, cell_file, head, None) for name, (file, cell_file, head) in NOTEBOOKS.items()]
    jobs += [(name, f"colab_t4_{name}.ipynb", "cell_suite.py", head, parts)
             for name, (parts, head) in SUITE_PARTS.items()]
    for name, file, cell_file, head, parts in jobs:
        src = compose(cell_file, EXTRA_WORKERS.get(name.split("_")[0] if parts else name))
        if parts:
            old = 'PARTS = ["lm", "vision", "speed", "data"]'
            assert old in src
            src = src.replace(old, f"PARTS = {parts}")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{name}.py").write_text(src, encoding="utf-8")
        nb = notebook(head + COMMON_PREP.replace("{cell}", "suite" if parts else name), src)
        out = nb_dir / file
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(out, len(nb["cells"][1]["source"]), "lines")


if __name__ == "__main__":
    main()
