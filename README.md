# memopro

[![PyPI](https://img.shields.io/pypi/v/memopro.svg?label=PyPI)](https://pypi.org/project/memopro/)
[![crates.io](https://img.shields.io/crates/v/memopro.svg?label=crates.io)](https://crates.io/crates/memopro)
[![docs.rs](https://img.shields.io/docsrs/memopro?label=docs.rs)](https://docs.rs/memopro)
[![CI](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#라이선스)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg?logo=python&logoColor=white)
![Status: beta](https://img.shields.io/badge/status-beta-yellow.svg)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E)
![Rust](https://img.shields.io/badge/Rust-000000?logo=rust&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-000000?logo=apple&logoColor=white)
![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?logo=nvidia&logoColor=white)
![LLM fine-tuning](https://img.shields.io/badge/LLM-LoRA%20fine--tuning-8A2BE2)
![Lossless](https://img.shields.io/badge/results-bit--identical-2E8B57)

**메모리보다 큰 언어 모델·비전 모델·데이터 작업을 정해진 메모리 상한 안에서, 결과 변화 없이 실행하는 Python·Rust 라이브러리**

한국어 · [English](README.en.md)

---

## 개요

8~16GB 노트북이나 작은 GPU에서는 모델이나 데이터가 메모리보다 크면 작업이 "메모리 부족(OOM)"으로 멈춥니다. 기존 해결책에는 각각 대가가 있습니다.

- **양자화**(4비트 등): 메모리를 줄이지만 모델의 출력과 정확도가 바뀝니다.
- **운영체제 스왑**: 결과는 같지만 사용량을 제한할 수 없고, 디스크에 대량으로 쓰며, 심하면 시스템 전체가 멈춥니다.
- **모델별 오프로딩 도구**: 특정 모델이나 작업에만 쓸 수 있습니다.

memopro는 이 문제를 **메모리 대신 시간을 쓰는 방식**으로 해결합니다. 지금 쓰지 않는 데이터를 메모리에서 비우고 필요할 때 되돌리되, 되돌리는 방법은 무손실만 사용합니다. 모델 가중치는 원본 파일에서 다시 읽고, 그 밖의 데이터는 무손실 압축합니다. 그 결과 다음이 보장됩니다.

- **결과 불변**: 생성·추론 출력은 메모리를 충분히 쓴 일반 실행과 비트 단위로 같습니다. 학습 손실은 예산과 관계없이 비트 단위로 같습니다(일반 PyTorch 학습과는 계산 순서 차이로 마지막 자리가 다를 수 있습니다).
- **메모리 상한**: 지킬 수 없는 예산은 시작 전에 거부합니다. 코드를 수정하지 않은 프로그램이 압축 후에도 상한을 넘으면 실행 중에 경고하고 초과량을 보고합니다.
- **디스크 쓰기 최소화**: 실행 중에 스왑 파일이나 임시 파일을 쓰지 않습니다. 디스크에 쓰는 것은 모델을 처음 한 번 내려받을 때뿐이고, 쓴 양을 알려 줍니다(`download="never"`로 끌 수 있음).

## 주요 기능

| 기능 | 설명 |
|---|---|
| 메모리보다 큰 LLM 미세조정 | 16비트 가중치 그대로 LoRA 학습. 예산에 맞는 batch 자동 선택 |
| 메모리보다 큰 LLM 생성 | 일반 생성과 같은 출력. 작은 초안 모델로 속도 향상 |
| 큰 비전 모델 추론 | Hugging Face 모델(DINOv2, ConvNeXt, CLIP·SigLIP 비전 부분 등)을 메모리보다 커도 실행 |
| 코드 수정 없는 메모리 상한 | `memopro run --budget 8GB script.py`로 NumPy·PyTorch 스크립트를 상한 안에서 실행 |
| 프로세스 메모리 상한 API | `memopro.enable(budget=...)` 한 줄로 프로세스 전체에 상한 적용 |
| 실행 전 진단과 예측 | 모델이 예산에 들어가는지, 상한을 걸면 얼마나 느려지는지 미리 계산 |
| 메모리보다 큰 CSV 정렬·병합 | `memopro.sort` / `memopro.join`: 중간 파일을 디스크에 쓰지 않고 상한 안에서 |
| 예산 런타임 | 큰 배열·텐서·모듈·옵티마이저를 예산 안에서 관리하는 Python·Rust·C API. `locked=True`로 실시간 버퍼를 RAM에 고정 |

## 주요 대상

| 분야 | 활용 예 |
|---|---|
| LLM 연구·개발 | 노트북이나 작은 GPU에서 7B급 모델의 16비트 LoRA 학습. 양자화 없는 기준선 실험과 재현성이 필요한 비교 실험 |
| 비전·멀티모달 | 큰 비전 인코더로 특징 추출·분류 추론 |
| 데이터 과학 | RAM보다 큰 NumPy 배열, scikit-learn·pandas 전처리 스크립트를 수정 없이 실행 |
| 교육·개인 학습 | 8GB 노트북, 무료 클라우드 GPU처럼 메모리가 작은 환경에서 큰 모델 실습 |
| 시스템·라이브러리 개발 | Rust·C 프로그램에 메모리 상한을 지키는 버퍼 런타임 내장 |

다음 경우에는 다른 도구가 더 적합합니다.
- 모델이 메모리에 다 들어가는 경우: Unsloth, mlx-tune 같은 전용 도구가 더 빠릅니다.
- 정확도 손실이 허용되는 경우: 4비트 양자화(QLoRA, llama.cpp)가 더 빠릅니다.

## 측정 결과

"개발 확인"으로 표시한 것을 빼면 모든 수치는 미리 정한 판정 기준으로 잰 값이고, 원시 결과는 [`docs/research/data/`](docs/research/data/)에 있습니다. 그림은 [`experiments/readme_figures.py`](experiments/readme_figures.py)가 결과 파일에서 직접 그립니다.

### 8GB Mac에서 메모리보다 큰 LLM 학습

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/fit-8gb-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/fit-8gb.svg" alt="8GB MacBook Air M1에서 Qwen2.5 1.5B, 3B, 7B의 16비트 LoRA: 3B부터 mlx-tune은 메모리 부족, memopro는 7B(가중치 15GB)까지 학습" width="720">
</picture>

가중치는 파일에 둔 채 필요한 층만 읽습니다. 상한을 바꿔도 학습 손실은 비트 단위로 같습니다.

<details><summary>표로 보기</summary>

| 모델 | 16비트 가중치 | mlx-tune | memopro |
|---|---|---|---|
| Qwen2.5-1.5B | 3.1 GB | 실행 (118.9 tok/s) | 실행 (45.4 tok/s) |
| Qwen2.5-3B | 6.2 GB | 메모리 부족 | 실행 (20.0 tok/s, 상한 1 GiB) |
| Qwen2.5-7B | 15.2 GB | 불러올 수 없음 | 실행 (16.4 tok/s, 1,024토큰) |

</details>

### 8GB Mac에서 메모리보다 큰 LLM 생성

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/generation-8gb-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/generation-8gb.svg" alt="토큰당 시간: 3B는 llama.cpp CPU 14.75초 대 memopro 1.09초, 7B는 초안 없이 8.95초에서 초안으로 2.37초" width="720">
</picture>

출력은 모델을 그냥 올렸을 때와 같습니다. llama.cpp는 Metal로 올리면 모두 메모리 부족이었고 CPU로만 돌았습니다.

<details><summary>표로 보기</summary>

| 경우 | 초/토큰 |
|---|---|
| 3B, llama.cpp (CPU) | 14.75 |
| 3B, memopro + int4 초안 | 1.09 |
| 7B, memopro (초안 없음) | 8.95 |
| 7B, memopro + 초안 | 2.37 (3.8배 빠름) |

</details>

### 프로세스 전체가 상한 안에

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ceiling-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ceiling.svg" alt="프로세스 최대 메모리 대 상한: T4 학습 91~93%, 데이터 프로그램 100~101%(허용 +32MiB), 1GB CSV 정렬 74%" width="720">
</picture>

상한은 memopro가 쥔 메모리만이 아니라 Python·torch까지 포함한 프로세스 전체에 걸립니다.

<details><summary>표로 보기</summary>

| 작업 (환경) | 상한 | 최대 메모리 |
|---|---|---|
| 3B 16비트 LoRA (Colab T4) | 6 / 8 GiB | 5.56 / 7.34 GiB |
| 7B 16비트 LoRA (Colab T4, 7B는 T4에 그냥 안 들어감) | 7 / 9 GiB | 6.36 / 8.37 GiB |
| 영상 묶음, 메모리 절반 (Colab) | 403 MiB | 407 MiB (허용 +32 MiB 안) |
| scikit-learn, 메모리 절반 (Colab) | 1,220 MiB | 1,222 MiB |
| 1GB CSV 정렬 (M1, 개발 확인) | 313 MiB | 233 MiB |

</details>

### 메모리 절반에서 그대로 실행, SSD에는 0바이트

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/half-memory-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/half-memory.svg" alt="메모리 절반에서 OS 스왑은 2,300MiB와 1,713MiB를 SSD에 썼고 memopro는 0바이트; 영상 작업은 스왑 62초 대 memopro 15초" width="720">
</picture>

코드를 고치지 않은 프로그램을 `memopro run --budget <절반>`으로 실행했습니다. 결과는 메모리를 다 쓴 실행과 같습니다. OS 스왑과 달리 SSD에 아무것도 쓰지 않습니다(억제율 100%).

<details><summary>표로 보기</summary>

| 작업 | SSD 쓰기: OS 스왑 / memopro | 시간: 메모리 전부 / OS 스왑 / memopro |
|---|---|---|
| 영상 묶음 (E027) | 2,300 MiB / 0 | 5.3 / 62.2 / 14.8초 |
| scikit-learn 분류 (E040) | 1,713 MiB / 0 | 11.7 / 25.5 / 27.0초 |

Linux 러너, cgroup 상한 = 일반 실행 최대 메모리의 1/2.

</details>

### 실시간 오디오: 다른 메모리가 스왑돼도 끊기지 않음

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/audio-locked-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/audio-locked.svg" alt="오디오 블록 렌더 시간 분포: 일반 버퍼는 마감 5.33ms를 3번 넘었고(최대 9.3ms) locked=True 버퍼는 0번(최대 2.5ms)" width="720">
</picture>

같은 메모리 공간에서 다른 프로세스가 약 8.5 GiB를 스왑하는 동안 60초를 렌더링했습니다. `Runtime.alloc(..., locked=True)`로 잡은 샘플 버퍼는 RAM에 고정되어 한 번도 끊기지 않았습니다.

<details><summary>표로 보기</summary>

| 경우 | 끊김 | 렌더 p99 / 최대 | 메이저 폴트 |
|---|---|---|---|
| 압박 없음, 일반 버퍼 | 0 | 0.06 / 0.24 ms | 0 |
| 압박 있음, 일반 버퍼 | 3 | 0.30 / 9.34 ms | 549 |
| 압박 있음, `locked=True` | 0 | 0.09 / 2.53 ms | 75 (파이썬 자체 코드 페이지) |

</details>

### 무손실 압축의 비율과 속도

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/compression-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/compression.svg" alt="자료 종류별 압축 비율: CSV 3.02배, int64 열쇠 2.07배, 영상 1.36배, float32 1.10배(압축 안 함), 무작위 바이트 압축 안 함" width="720">
</picture>

메모리가 모자라면 쓰지 않는 버퍼를 메모리 안에서 무손실로 압축합니다(디스크에 쓰지 않음). 거의 줄지 않는 자료는 압축하지 않고 그대로 둡니다. 모든 버퍼가 비트 단위로 같게 돌아왔습니다. Apple M1에서는 같은 비율에 속도가 약 두 배였습니다(CSV 압축 0.98 GB/s, 해제 3.23 GB/s).

### 다른 언어에서 부르기

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ffi-dark.svg">
  <img src="https://raw.githubusercontent.com/imhyensuk/memopro/main/docs/images/ffi.svg" alt="런타임 호출 비용: C ABI는 Rust와 같고 Python은 호출당 약 5배; 대량 복사는 세 언어 모두 memcpy 속도" width="720">
</picture>

C 프로그램은 `memopro-c`(`include/memopro.h`)로 Rust와 같은 비용에 런타임을 씁니다. 세 언어가 같은 바이트를 냈습니다.

## 설치

```bash
pip install memopro            # 런타임, memopro run, enable, 진단
pip install "memopro[llm]"     # + finetune, generate (PyTorch, Transformers, PEFT)
```

0.2.0 베타를 쓰려면 `pip install --pre memopro`(또는 `"memopro[llm]==0.2.0b4"`)로 설치합니다. 0.1.x에서 바뀐 점은 [CHANGELOG](CHANGELOG.md)의 "0.1.x에서 옮겨 오기"에 있습니다.

| 추가 옵션 | 포함 |
|---|---|
| `memopro[numpy]` | NumPy |
| `memopro[torch]` | PyTorch, NumPy |
| `memopro[hf]` | PyTorch, Transformers, Accelerate, safetensors |
| `memopro[llm]` | `hf` + PEFT, torchao |
| `memopro[notebook]` | IPython (Jupyter 매직) |

**요구 사항**: Python 3.11 이상, PyTorch 2.4 이상(AI 기능).

| 플랫폼 | 지원 |
|---|---|
| macOS, Apple silicon | 전체 기능 (Apple GPU) |
| Linux x86_64·aarch64 + NVIDIA GPU | 전체 기능 (CUDA) |
| Linux CPU | GPU 기능을 제외한 전체 기능 |
| Windows x86_64 | CPU 기능(LLM 학습·생성, sort/join, 런타임). `memopro run`의 투명 페이징과 `locked=True`는 미지원 |

PyPI는 macOS arm64, Linux x86_64·aarch64, Windows x86_64 wheel을 제공합니다. 그 밖의 환경은 Rust 툴체인으로 소스에서 빌드됩니다. Rust 프로젝트는 `cargo add memopro`로 추가합니다.

## 명령어와 API

### 명령줄

| 명령 | 설명 |
|---|---|
| `memopro doctor` | 장치(GPU)·호스트(RAM)·디스크별 메모리와 상한 (`--json`, `--no-devices`, `--budget`) |
| `memopro run [옵션] script.py [인자]` | 스크립트를 수정 없이 메모리 상한 안에서 실행 |
| `memopro --version` | 버전 |

`memopro run`의 주요 옵션: `--budget`(상한), `--dry-run`(설정만 출력), `--report-json PATH`(결과 보고서), `--no-numpy` / `--no-torch`(해당 자동 처리 끔).

### Python API

| 함수 | 설명 |
|---|---|
| `memopro.finetune(model, texts, budget=, batch=, seq_len=, epochs=, lr=, rank=)` | 메모리보다 큰 LLM의 LoRA 학습. `FinetuneResult`(`model`, `adapter`, `losses`, `batch` 등) 반환. 직접 불러온 PyTorch 모델(Unsloth `FastLanguageModel`, Transformers)을 `tokenizer=`와 함께 주면 그 모델 그대로(Unsloth면 Unsloth의 LoRA 층과 속도로) 학습하고, memopro는 프로세스 상한만 확인함. `mlx_lm.load`로 불러온 모델을 `tokenizer=`와 함께 주면 mlx-lm의 LoRA 층과 손실로 학습하며, 결과는 mlx-lm 학습과 같음 |
| `memopro.generate(model, prompt, budget=, draft="auto", max_new_tokens=, streamer=, stopping_criteria=)` | 메모리보다 큰 LLM의 생성. 디스크에 있는 같은 계열의 작은 모델을 초안으로 자동 사용. `streamer`(예: Transformers `TextIteratorStreamer`)와 `stopping_criteria`는 `model.generate`에 그대로 넘어가며, 스트리머가 받는 텍스트는 반환값과 같음. `mlx_lm.load`로 불러온 모델을 `tokenizer=`와 함께 주면 mlx-lm이 그대로 생성하고, memopro는 상한에 맞지 않는 층만 토큰마다 다시 읽음(출력은 mlx-lm과 같음, `pip install memopro[mlx]`). |
| `memopro.enable(budget=)` / `memopro.disable()` | 프로세스 전체 메모리 상한. `Session.estimate()`, `Session.measured()` |
| `memopro.load(model_id, budget=, **from_pretrained)` | Hugging Face 모델을 저장된 그대로 불러오기(정밀도를 바꾸지 않음). 예산에 안 들어가면 흘려 쓰기로 불러옴. `dtype=`·`quantization_config=`를 직접 주면 그 모델이 원본이 되어 준 그대로 불러옴. 디스크에는 아무것도 쓰지 않음 |
| `memopro.sort(path, key, out=, budget=)` / `memopro.join(left, right, on, out=, budget=)` | 메모리보다 큰 CSV를 한 열로 정렬하거나 병합(정수면 숫자 순서, 아니면 바이트 순서). 상한 안에서 돌고 중간 파일을 디스크에 쓰지 않음(원본을 필요한 만큼 다시 읽고, 바뀌었으면 오류). `out`은 파일 경로나 바이트 조각을 받는 함수. RFC 4180 CSV(머리글, `,` 구분, 따옴표 필드, `\n`·`\r\n` 줄 끝) |
| `memopro.doctor()` | CLI와 같은 메모리·상한 보고 |
| `memopro.configure(...)` / `memopro.using(...)` | 전역 설정 / 블록 단위 설정 |

0.2.0에서 `check`, `train_session`, `optimize`, `hibernate`, `census`, `elastic`, 기법 등록 API, CLI `memopro check`가 공개 API에서 빠졌습니다. 쓰면 무엇으로 바꾸면 되는지 알려 주는 오류가 납니다. 학습은 `finetune`이나 `memopro.enable()` 안에서 평소 코드로 하면 됩니다.

### 런타임 API

| 이름 | 설명 |
|---|---|
| `memopro.rt.Runtime(budget=)` | 예산 안에서 버퍼를 관리하는 런타임 |
| `Runtime.load_npy`, `.array`, `.add_file`, `.alloc` | 파일·새 버퍼 등록. `Buffer.view()`로 복사 없는 NumPy 뷰. `alloc(..., locked=True)`는 버퍼를 RAM에 고정(압축·스왑 없음, 실시간 오디오 버퍼처럼 기다리면 안 되는 메모리용, macOS·Linux) |
| `Runtime.adopt(obj)` | 텐서·모듈·옵티마이저·배열을 담은 객체를 런타임에 맡김 |
| `memopro.rt.torch.stream_model(name, budget=, device=, model_class=)` | 가중치를 파일에 둔 채 Hugging Face 모델 실행 |
| `memopro.rt.torch.draft_model(name, target=, device=)` | 생성 속도용 초안 모델 (Apple GPU: int4, NVIDIA GPU: 16비트) |
| `memopro.rt.process_footprint()` | 현재 프로세스 메모리 사용량 |

### 예산 형식과 설정

예산은 모든 함수, `--budget`, 설정 파일 `memopro.toml`(또는 `$MEMOPRO_CONFIG`), 환경 변수 `MEMOPRO_BUDGET`에서 같은 형식을 씁니다. 다른 설정도 `MEMOPRO_<이름>` 환경 변수로 줄 수 있습니다.

| 형식 | 의미 |
|---|---|
| `"auto"` | 이 기계 메모리의 80% (기본값) |
| `"6GB"` | 정확히 6GB. 128GB 기계에서도, 16GB 기계에서도 그대로 상한이 됩니다 |
| `"50%"`, `0.5` | 이 기계 메모리의 절반 |
| `"device=8GB,host=24GB"` 또는 `{"device": "8GB", "host": "24GB"}` | NVIDIA GPU 메모리와 RAM을 따로 지정 (Apple silicon은 하나로 공유) |

정한 값이 지금 비어 있는 메모리보다 크면 상한은 그대로 두고, 다른 프로그램이 쓰는 몫 때문에 스왑이 생길 수 있다고 알려 줍니다. 0.1.x의 `"-2GB"`, `"2GB..6GB"`, `"6GB!"`, `budget_basis`, `headroom`은 0.2.0에서 빠졌고, 쓰면 새 표기법을 알려 주는 오류가 납니다.

## 빠른 시작

### 작은 언어 모델 학습과 생성

**사전 요구 사항**
- `pip install "memopro[llm]"`
- 인터넷 연결: 처음 실행할 때 Hugging Face에서 공개 모델 Qwen2.5-0.5B(약 1GB)를 한 번 내려받습니다. 토큰은 필요 없습니다. 쓴 양은 안내 메시지로 표시되고, 이후 실행은 내려받지 않습니다.
- 장치는 자동으로 선택됩니다(NVIDIA GPU → Apple GPU → CPU). Apple silicon 맥에서 약 1~2분이 걸립니다.

```python
import memopro

texts = [
    "memopro runs models that are larger than the memory of the machine.",
    "It reads each layer's weights from the file only while that layer runs.",
] * 8

# the first run downloads the model once (about 1 GB); later runs read it from the cache
r = memopro.finetune("Qwen/Qwen2.5-0.5B-Instruct", texts, budget="1GiB", seq_len=64)
print("losses:", [round(x, 3) for x in r.losses])
print(memopro.generate(r.model, "What does memopro do?", max_new_tokens=30))
```

`budget`보다 큰 모델도 같은 코드로 학습합니다. 모델 이름과 예산만 바꾸면 됩니다(예: `"Qwen/Qwen2.5-7B-Instruct"`, `budget="2.5GiB"`).

### 기존 스크립트를 상한 안에서 실행

**사전 요구 사항**: `pip install memopro numpy`. macOS 또는 Linux가 필요합니다.

아래 스크립트는 보통 약 2.4GB를 씁니다(macOS 측정). 파일로 저장한 뒤 memopro로 실행하면 1GB 안에서 같은 결과를 냅니다.

```python
# big_job.py: an ordinary NumPy program, no memopro code in it
import numpy as np

x = (np.arange(200_000_000, dtype=np.float32) % 1000) / 1000
y = np.sqrt(x) * 2
print("sum:", float(y.sum(dtype=np.float64)))
```

```bash
python big_job.py                          # 최대 약 2.4GB
memopro run --budget 1GB big_job.py        # 최대 약 1GB, 같은 결과
```

### 사용 가능한 메모리 확인

```bash
memopro doctor
```

## 고급 사용법과 벤치마크

### 학습 속도: `batch="auto"`

```python
r = memopro.finetune("Qwen/Qwen2.5-0.5B-Instruct", texts, budget="2.5GiB", batch="auto")
print(r.batch)                           # 고른 batch
r.adapter.save_pretrained("my-lora")     # 표준 PEFT 어댑터
```

가중치는 한 단계에 한 번만 읽히므로 batch가 클수록 토큰당 시간이 줄어듭니다. 결과는 같은 batch끼리 예산과 관계없이 같습니다.

### 생성 속도: 초안 모델

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

작은 모델이 토큰을 제안하고 큰 모델이 검증합니다. 출력은 큰 모델만으로 생성한 결과와 같습니다. 초안은 Apple GPU(int4)와 NVIDIA GPU(16비트)에서 쓸 수 있고, 같은 토크나이저를 쓰는 모델이어야 합니다. NVIDIA GPU에서 자동 초안은 초안이 차지하는 GPU 캐시보다 얻는 이득이 클 때만 씁니다. 모델이 GPU보다 훨씬 큰 경우입니다(T4에서 3B·7B는 초안 없이가 더 빨랐습니다. 7B: 1.55초/토큰, 초안 사용 시 7.3초/토큰).

`draft`를 주지 않으면(기본값 `"auto"`) 이 디스크에 이미 있는 모델 가운데 같은 토크나이저를 쓰고 크기가 약 1/5인 모델을 골라 초안으로 씁니다. 찾는 곳은 세 군데입니다.

1. `model_dirs` 설정에 적은 폴더(모델 폴더 또는 모델 폴더들을 담은 폴더)
2. 폴더에서 불러온 모델이라면 그 옆 폴더들
3. Hugging Face 캐시(있을 때만)

```python
memopro.configure(model_dirs=["~/models", "/Volumes/SSD/llm"])  # 또는 MEMOPRO_MODEL_DIRS=~/models:/Volumes/SSD/llm
```

그런 모델이 없고 모델이 예산보다 커서 매 토큰 흘려 읽어야 한다면(Apple GPU), 모델 자신의 int4 사본을 메모리에만 만들어 초안으로 씁니다(디스크에 쓰지 않음, 출력은 같음). 초안 때문에 새로 내려받는 것은 없습니다. 맞는 모델이 없거나 예산이 초안까지 담지 못하면 초안 없이 생성하고(출력은 같습니다), 어느 정도 예산이면 초안을 쓸 수 있는지 알려 줍니다. `draft=None`이면 초안을 쓰지 않습니다.

### 비전 모델

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).pooler_output
```

`model_class=transformers.SiglipVisionModel`처럼 CLIP·SigLIP 체크포인트의 비전 부분만 실행할 수도 있습니다.

### 코드 안에서 상한과 예측

```python
s = memopro.enable(budget="8GB")                    # 프로세스 전체 상한
print(s.estimate(sample, total="12GB", passes=3))   # 실행 전: 예상 추가 시간
...                                                 # NumPy / PyTorch 코드
print(s.measured())                                 # 실행 후: 실제 시간, 최대 메모리, 초과량
```

### 런타임에 직접 맡기기

```python
from memopro.rt import Runtime

rt = Runtime(budget="2GB")
big = rt.load_npy("big.npy")                 # 파일 원본: 필요하면 다시 읽음(해시 검증)
work = rt.array((50_000, 4_096), "float32")  # 새 버퍼: 쓰지 않는 동안 무손실 압축
with work.view(write=True) as a:             # 사용하는 동안만 메모리에 올라오는 NumPy 뷰
    a[:] = 1.0
```

### 메모리보다 큰 CSV 정렬·병합

```python
import hashlib, memopro

memopro.sort("events.csv", key="user_id", out="sorted.csv", budget="8GB")

h = hashlib.sha256()  # 출력을 파일 대신 함수로 받기
memopro.join("events.csv", "users.csv", on="user_id", out=h.update, budget="8GB")
```

중간 파일을 쓰지 않고, 필요한 만큼 원본을 다시 읽습니다. 블록마다 첫 읽기의 CRC와 맞춰 보므로 도중에 파일이 바뀌면 틀린 결과 대신 오류가 납니다. 한 번에 담은 행은 상한에 닿으면 무손실 압축됩니다. `out`은 파일 경로이거나 출력 조각을 받는 함수입니다. RFC 4180 CSV(머리글, `,` 구분, 쉼표·줄바꿈·`""`를 담은 따옴표 필드, `\n`·`\r\n` 줄 끝)를 받습니다. 열쇠는 파일의 모든 값이 정수면 숫자로, 아니면 글자(바이트 순서, `LC_ALL=C sort`와 같음)로 비교합니다. 잘못된 입력은 잘못 읽지 않고 거부합니다.

### 실시간 버퍼 고정

```python
from memopro.rt import Runtime

rt = Runtime(budget="1GB")
samples = rt.array((48_000 * 60,), "int16", locked=True)  # 압축·스왑되지 않음
```

### 벤치마크

| 작업 | 환경 | 결과 |
|---|---|---|
| Qwen2.5-7B LoRA 학습 | Apple M1, 8GB | 예산 2.5GiB, `batch=4`에서 15.5토큰/초 |
| Mistral-7B LoRA 학습 | NVIDIA T4, 16GB | 예산 4GiB, 약 43토큰/초 |
| Qwen2.5-7B 생성 (`draft` 사용) | Apple M1, 8GB | 8.95 → 2.37초/토큰, 출력 동일 |
| Qwen2.5-3B LoRA 학습 | Apple M1, 8GB | 20토큰/초 (mlx-tune은 메모리 부족으로 실행 불가) |
| Qwen2.5-3B LoRA 학습, `mlx_lm.load` 모델 | Apple M1, 8GB | 상한 2.5GiB에서 22.8토큰/초, 손실·어댑터가 mlx-lm 학습과 비트 동일 (mlx-lm 그대로는 1.8토큰/초, 스왑 +0.9GB; mlx-tune은 메모리 부족) |
| 1.5B~3.8B 모델 (Qwen2.5, SmolLM2, Phi-3.5) | NVIDIA T4, 16GB | 메모리를 충분히 쓴 실행 대비 학습 1.00~1.03배, 생성 1.06~1.35배 시간 |
| NumPy·scikit-learn·CSV 처리, 필요 메모리의 50~75% 상한 | Linux | 결과 동일, 1.1~3.4배 시간 |
| 같은 3B LoRA, 모델이 GPU에 들어가는 크기 | NVIDIA T4 | Unsloth 0.45초/스텝 대 memopro 5.1초/스텝 (Unsloth가 약 11배 빠름) |

검증한 모델: Qwen2.5(1.5B~14B), SmolLM2, Phi-3.5, Mistral-7B, DINOv2, ConvNeXtV2, ResNet-152. 원시 결과는 [`docs/research/data/`](docs/research/data/)에 있습니다.

### 한계

- 메모리를 크게 넘는 모델은 실행되지만 저장 장치 읽기 속도에 묶여 매우 느립니다. 예를 들어 NVIDIA T4에서 14B 16비트 LoRA는 약 1토큰/초입니다.
- 모델이 메모리에 들어가는 크기에서는 전용 도구가 훨씬 빠릅니다(T4 3B: Unsloth가 약 11배, M1 1.5B: mlx-tune이 2.6배). memopro의 장점은 들어가지 않는 크기를 결과를 바꾸지 않고 상한 안에서 돌리는 데 있습니다.
- 코드를 고치지 않은 무작위 접근 작업(pandas 데이터프레임 조작, Polars 등)은 `memopro run`에서 크게 느려지거나 상한을 지키지 못할 수 있습니다. 큰 CSV의 정렬·병합에는 `memopro.sort`/`memopro.join`을 쓰세요.
- CUDA에서 학습 손실은 상한과 관계없이 같지만, 일반 PyTorch 학습과는 2단계부터 마지막 자리가 다를 수 있습니다.
- `locked=True`는 데이터 버퍼만 고정합니다. 프로그램 코드까지 고정하려면 프로그램이 직접 `mlockall`을 써야 합니다.
- Apple GPU에서는 `finetune`/`stream_model`을 다른 MPS 연산보다 먼저 호출해야 메모리 설정이 적용됩니다.
- 베타 버전이라 API가 바뀔 수 있습니다. 변경 내역은 [CHANGELOG](CHANGELOG.md)에 있습니다.

## 기여와 라이선스

### 기여

버그 보고와 제안은 [Issues](https://github.com/imhyensuk/memopro/issues)에, 코드 기여는 Pull Request로 보내 주세요. 큰 변경은 먼저 이슈로 논의하는 것을 권장합니다.

개발 환경:

```bash
git clone https://github.com/imhyensuk/memopro.git && cd memopro
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release   # Rust 확장 빌드
```

Pull Request 전에 다음이 통과해야 합니다.

```bash
.venv/bin/pytest -q
.venv/bin/ruff check python tests && .venv/bin/ruff format --check python tests
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p memopro
```

- 새 기능과 버그 수정에는 시험을 함께 추가합니다.
- `import memopro`는 부작용이 없어야 하며 torch·numpy를 불러오지 않습니다.
- 무손실 원칙을 지킵니다. 결과를 바꾸는 방식은 기본값이 될 수 없습니다.

### 인용

연구에 사용한 경우 저장소의 **"Cite this repository"**([`CITATION.cff`](CITATION.cff))로 인용해 주세요.

### 라이선스

[MIT](LICENSE-MIT) 또는 [Apache-2.0](LICENSE-APACHE) 중 선택할 수 있습니다. 기여한 코드도 같은 이중 라이선스로 배포됩니다.
