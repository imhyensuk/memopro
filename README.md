# memopro

**메모리보다 큰 작업을 결과 변화 없이 실행하는 Python·Rust 라이브러리**

[![PyPI](https://img.shields.io/pypi/v/memopro.svg)](https://pypi.org/project/memopro/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#라이선스)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

한국어 · [English](README.en.md)

memopro는 기기의 메모리보다 큰 언어 모델, 비전 모델, 데이터 처리 작업을 정해진 메모리 상한 안에서 실행합니다. 양자화나 근사 없이 동작하므로 출력과 학습 손실은 메모리를 충분히 쓴 실행과 비트 단위로 같고, 그 대가로 실행 시간이 늘어납니다.

```bash
pip install "memopro[llm]"
```

```python
import memopro

# 7B 모델(가중치 14GB)을 2GiB 메모리 안에서 16비트 LoRA로 학습
r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="2GiB")
print(memopro.generate(r.model, "Hello"))
```

## 주요 기능

- **메모리보다 큰 언어 모델의 미세조정과 생성**: 16비트 가중치를 그대로 쓰는 LoRA 학습과 텍스트 생성
- **큰 비전 모델 추론**: Hugging Face 모델을 메모리보다 커도 그대로 실행
- **코드 수정 없는 메모리 상한**: `memopro run --budget 8GB script.py`로 기존 NumPy·PyTorch 스크립트를 상한 안에서 실행
- **실행 전 진단과 예측**: 모델이 예산에 들어가는지, 상한을 걸면 얼마나 느려지는지 미리 계산
- **Python, Rust, C에서 사용 가능**

모든 기능은 다음을 보장합니다.

- **결과 불변**: 무손실 방식만 사용하며, 출력은 일반 실행과 같습니다.
- **메모리 상한**: 지킬 수 없는 예산은 시작 전에 거부합니다. 코드를 수정하지 않은 프로그램이 압축 후에도 상한을 넘으면 실행 중에 경고하고 초과량을 보고합니다.
- **디스크 쓰기 없음**: 스왑 파일이나 캐시 파일을 만들지 않습니다.

## 사용 예

### 언어 모델 미세조정

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-7B-Instruct", texts,
    budget="2.5GiB",   # 메모리 상한
    batch="auto",      # 상한에 들어가는 가장 큰 batch 선택
)
r.adapter.save_pretrained("my-lora")   # 표준 PEFT 어댑터
```

장치는 자동으로 선택됩니다(NVIDIA GPU, Apple GPU, CPU 순). 같은 데이터는 예산과 관계없이 같은 손실을 냅니다.

### 텍스트 생성

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

`draft`에 작은 모델을 지정하면 작은 모델이 토큰을 먼저 제안하고 큰 모델이 이를 검증합니다. 출력은 큰 모델만으로 생성한 결과와 같습니다.

### 비전 모델 추론

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).pooler_output
```

`model_class`로 CLIP·SigLIP 체크포인트의 비전 부분(`CLIPVisionModel`, `SiglipVisionModel`)만 실행할 수도 있습니다.

### 기존 스크립트를 수정 없이 실행

```bash
memopro run --budget 8GB analysis.py   # 큰 NumPy 배열을 상한 안에서 무손실 압축
memopro run --budget 50% train.py      # 측정한 여유 메모리의 절반 안에서 실행
memopro run --dry-run train.py         # 적용할 설정만 출력
```

### 코드에서 상한 설정

```python
import memopro

s = memopro.enable(budget="8GB")                    # 프로세스 전체의 메모리 상한
print(s.estimate(sample, total="12GB", passes=3))   # 예상 추가 시간
...                                                 # NumPy / PyTorch 코드
print(s.measured())                                 # 실제 시간과 최대 메모리
```

### 진단

```bash
memopro doctor                                                      # 장치별 사용 가능 메모리
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # 모델이 예산에 들어가는지
```

자세한 사용법은 [사용 가이드](docs/guide/README.md)를 참고하세요.

### 예산 형식

모든 함수, `--budget`, `memopro.toml`, 환경 변수 `MEMOPRO_BUDGET`은 같은 형식을 씁니다.

| 형식 | 의미 |
|---|---|
| `"auto"` | 측정한 여유 메모리 − 10% (기본값) |
| `"6GB"`, `"50%"`, `0.5` | 크기, 또는 여유 메모리 대비 비율 |
| `"-2GB"` | 2GB를 남기고 나머지 |
| `"2GB..6GB"` | 최대 6GB, 2GB를 확보할 수 없으면 시작하지 않음 |
| `"6GB!"` | 측정값과 관계없이 6GB (스왑 가능성 있음) |

## 벤치마크

| 작업 | 환경 | 결과 |
|---|---|---|
| Qwen2.5-7B LoRA 학습 | Apple M1, 8GB | 예산 2.5GiB, `batch=4`에서 15.5토큰/초 |
| Mistral-7B LoRA 학습 | NVIDIA T4, 16GB | 예산 4GiB, 약 43토큰/초 |
| Qwen2.5-7B 생성 (`draft` 사용) | Apple M1, 8GB | 8.95 → 2.37초/토큰, 출력 동일 |
| Qwen2.5-3B LoRA 학습 | Apple M1, 8GB | 20토큰/초 (mlx-tune은 메모리 부족으로 실행 불가) |
| 1.5B~3.8B 모델 (Qwen2.5, SmolLM2, Phi-3.5) | NVIDIA T4, 16GB | 메모리를 충분히 쓴 실행 대비 학습 1.00~1.03배, 생성 1.06~1.35배 시간 |
| NumPy·scikit-learn·CSV 처리, 필요 메모리의 50~75% 상한 | Linux | 결과 동일, 1.1~3.4배 시간 |

검증한 모델: Qwen2.5(1.5B~14B), SmolLM2, Phi-3.5, Mistral-7B, DINOv2, ConvNeXtV2, ResNet-152. 원시 결과는 [`docs/research/data/`](docs/research/data/)에 있습니다.

## 적용 범위와 한계

- **적합한 경우**: 모델이나 데이터가 메모리를 조금 넘는 경우, 결과가 원래 실행과 같아야 하는 경우
- **메모리에 다 들어가는 모델**: Unsloth, mlx-tune 같은 전용 도구가 더 빠릅니다.
- **메모리를 크게 넘는 모델**: 실행은 되지만 저장 장치 읽기 속도에 묶여 매우 느립니다. 예를 들어 NVIDIA T4에서 14B 16비트 LoRA는 약 1토큰/초입니다.
- **정확도 손실이 허용되는 경우**: 4비트 양자화(QLoRA, llama.cpp)가 더 빠릅니다.
- **무작위 접근이 많은 데이터 작업**(정렬, 조인, Polars 등): 결과는 같지만 크게 느려질 수 있습니다.
- **Apple GPU**: `finetune`/`stream_model`은 다른 MPS 연산보다 먼저 호출해야 메모리 설정이 적용됩니다. 늦게 호출하면 경고가 표시됩니다.
- **알파 버전**: API가 바뀔 수 있습니다. 변경 내역은 [CHANGELOG](CHANGELOG.md)에 있습니다.

## 설치

| 명령 | 포함 기능 |
|---|---|
| `pip install memopro` | `memopro run`, `enable`, 런타임, 진단 |
| `pip install "memopro[llm]"` | `finetune`, `generate` (PyTorch, Transformers, PEFT 포함) |
| `pip install "memopro[notebook]"` | Jupyter 매직 (`%hibernate`, `%wake`) |

| 플랫폼 | 지원 |
|---|---|
| macOS, Apple silicon | 전체 기능 |
| Linux + NVIDIA GPU | 전체 기능 |
| Linux CPU | GPU 기능을 제외한 전체 기능 |
| Windows | 기본 기능 |

Python 3.11 이상, PyTorch 2.4 이상이 필요합니다. PyPI는 macOS arm64, Linux x86_64·aarch64 wheel을 제공합니다. Rust는 [crates.io](https://crates.io/crates/memopro), C는 [`crates/memopro-c`](crates/memopro-c)를 사용합니다.

## 동작 원리

memopro는 사용하지 않는 데이터를 메모리에서 비우고 필요할 때 되돌립니다. 언어 모델은 계산할 층의 가중치만 원본 파일에서 읽고 사용 후 해제합니다. 그 밖의 데이터는 바이트 단위로 복원되는 무손실 압축을 사용합니다. 근사나 양자화를 쓰지 않으므로 결과는 그대로이고 실행 시간만 늘어납니다.

## 개발

```bash
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest -q
cargo test -p memopro
```

## 인용

연구에 사용한 경우 저장소의 **"Cite this repository"**([`CITATION.cff`](CITATION.cff))로 인용해 주세요.

## 라이선스

[MIT](LICENSE-MIT) 또는 [Apache-2.0](LICENSE-APACHE) 중 선택.
