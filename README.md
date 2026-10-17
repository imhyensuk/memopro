# memopro

[![PyPI](https://img.shields.io/pypi/v/memopro.svg?label=PyPI)](https://pypi.org/project/memopro/)
[![crates.io](https://img.shields.io/crates/v/memopro.svg?label=crates.io)](https://crates.io/crates/memopro)
[![docs.rs](https://img.shields.io/docsrs/memopro?label=docs.rs)](https://docs.rs/memopro)
[![CI](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/imhyensuk/memopro/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#라이선스)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg?logo=python&logoColor=white)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

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

- **결과 불변**: 출력과 학습 손실이 메모리를 충분히 쓴 실행과 비트 단위로 같습니다.
- **메모리 상한**: 지킬 수 없는 예산은 시작 전에 거부합니다. 코드를 수정하지 않은 프로그램이 압축 후에도 상한을 넘으면 실행 중에 경고하고 초과량을 보고합니다.
- **디스크 쓰기 없음**: 스왑 파일이나 캐시 파일을 만들지 않습니다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| 메모리보다 큰 LLM 미세조정 | 16비트 가중치 그대로 LoRA 학습. 예산에 맞는 batch 자동 선택 |
| 메모리보다 큰 LLM 생성 | 일반 생성과 같은 출력. 작은 초안 모델로 속도 향상 |
| 큰 비전 모델 추론 | Hugging Face 모델(DINOv2, ConvNeXt, CLIP·SigLIP 비전 부분 등)을 메모리보다 커도 실행 |
| 코드 수정 없는 메모리 상한 | `memopro run --budget 8GB script.py`로 NumPy·PyTorch 스크립트를 상한 안에서 실행 |
| 프로세스 메모리 상한 API | `memopro.enable(budget=...)` 한 줄로 프로세스 전체에 상한 적용 |
| 실행 전 진단과 예측 | 모델이 예산에 들어가는지, 상한을 걸면 얼마나 느려지는지 미리 계산 |
| 예산 런타임 | 큰 배열·텐서·모듈·옵티마이저를 예산 안에서 관리하는 Python·Rust·C API |

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

## 설치

```bash
pip install memopro            # 런타임, memopro run, enable, 진단
pip install "memopro[llm]"     # + finetune, generate (PyTorch, Transformers, PEFT)
```

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
| Windows | 기본 기능 (`memopro run`의 페이징 미지원) |

PyPI는 macOS arm64, Linux x86_64·aarch64 wheel을 제공합니다. 그 밖의 환경은 Rust 툴체인으로 소스에서 빌드됩니다. Rust 프로젝트는 `cargo add memopro`로 추가합니다.

## 명령어와 API

### 명령줄

| 명령 | 설명 |
|---|---|
| `memopro doctor` | 장치(GPU)·호스트(RAM)·디스크별 사용 가능 메모리 (`--json`, `--no-devices`) |
| `memopro check MODEL` | 모델이 예산에 들어가는지와 적합한 설정 (`--goal infer\|train\|both`, `--batch-size`, `--seq-len`, `--optimizer`) |
| `memopro run [옵션] script.py [인자]` | 스크립트를 수정 없이 메모리 상한 안에서 실행 |
| `memopro --version` | 버전 |

`memopro run`의 주요 옵션: `--budget`(상한), `--dry-run`(설정만 출력), `--report-json PATH`(결과 보고서), `--no-numpy` / `--no-torch`(해당 자동 처리 끔), `--quality lossless|high|balanced|low`(불러오기 품질 하한).

### Python API

| 함수 | 설명 |
|---|---|
| `memopro.finetune(model, texts, budget=, batch=, seq_len=, epochs=, lr=, rank=)` | 메모리보다 큰 LLM의 LoRA 학습. `FinetuneResult`(`model`, `adapter`, `losses`, `batch` 등) 반환 |
| `memopro.generate(model, prompt, budget=, draft=, max_new_tokens=)` | 메모리보다 큰 LLM의 생성 |
| `memopro.enable(budget=)` / `memopro.disable()` | 프로세스 전체 메모리 상한. `Session.estimate()`, `Session.measured()` |
| `memopro.load(model_id, budget=, quality=)` | 예산에 맞는 형식을 골라 Hugging Face 모델 불러오기 |
| `memopro.check(...)` / `memopro.doctor()` | CLI와 같은 진단 |
| `memopro.train_session(model, optimizer)` | 예산에 맞춰 마이크로배치·체크포인팅을 고르는 정확한 학습 |
| `memopro.optimize(model, goal=)` | 이미 가진 모델을 예산에 맞춤 |
| `memopro.configure(...)` / `memopro.using(...)` | 전역 설정 / 블록 단위 설정 |

### 런타임 API

| 이름 | 설명 |
|---|---|
| `memopro.rt.Runtime(budget=)` | 예산 안에서 버퍼를 관리하는 런타임 |
| `Runtime.load_npy`, `.array`, `.add_file`, `.alloc` | 파일·새 버퍼 등록. `Buffer.view()`로 복사 없는 NumPy 뷰 |
| `Runtime.adopt(obj)` | 텐서·모듈·옵티마이저·배열을 담은 객체를 런타임에 맡김 |
| `memopro.rt.torch.stream_model(name, budget=, device=, model_class=)` | 가중치를 파일에 둔 채 Hugging Face 모델 실행 |
| `memopro.rt.torch.draft_model(name, target=)` | 생성 속도용 int4 초안 모델 |
| `memopro.rt.process_footprint()` | 현재 프로세스 메모리 사용량 |

### Jupyter

```text
%load_ext memopro
%hibernate 변수     # 객체를 잠시 메모리에서 내림
%wake 변수          # 되돌림
%memopro status
```

### 예산 형식과 설정

예산은 모든 함수, `--budget`, 설정 파일 `memopro.toml`(또는 `$MEMOPRO_CONFIG`), 환경 변수 `MEMOPRO_BUDGET`에서 같은 형식을 씁니다. 다른 설정도 `MEMOPRO_<이름>` 환경 변수로 줄 수 있습니다.

| 형식 | 의미 |
|---|---|
| `"auto"` | 측정한 여유 메모리 − 10% (기본값) |
| `"6GB"`, `"50%"`, `0.5` | 크기, 또는 여유 메모리 대비 비율 |
| `"-2GB"` | 2GB를 남기고 나머지 |
| `"2GB..6GB"` | 최대 6GB, 2GB를 확보할 수 없으면 시작하지 않음 |
| `"6GB!"` | 측정값과 관계없이 6GB (스왑 가능성 있음) |
| `{"device": "80%", "host": "-2GB"}` | 메모리 종류별 지정 |

## 빠른 시작

### 작은 언어 모델 학습과 생성

**사전 요구 사항**
- `pip install "memopro[llm]"`
- 인터넷 연결: 처음 실행할 때 Hugging Face에서 공개 모델 Qwen2.5-0.5B(약 1GB)를 내려받습니다. 토큰은 필요 없습니다.
- memopro는 디스크에 쓰지 않으므로 모델을 자동으로 받지 않습니다. 아래 코드처럼 먼저 내려받아야 합니다.
- 장치는 자동으로 선택됩니다(NVIDIA GPU → Apple GPU → CPU). Apple silicon 맥에서 약 1~2분이 걸립니다.

```python
from huggingface_hub import snapshot_download

import memopro

# memopro never writes to disk, so download the model once first (about 1 GB)
path = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")

texts = [
    "memopro runs models that are larger than the memory of the machine.",
    "It reads each layer's weights from the file only while that layer runs.",
] * 8

r = memopro.finetune(path, texts, budget="1GiB", seq_len=64)
print("losses:", [round(x, 3) for x in r.losses])
print(memopro.generate(r.model, "What does memopro do?", max_new_tokens=30))
```

`budget`보다 큰 모델도 같은 코드로 학습합니다. 모델 이름과 예산만 바꾸면 됩니다(예: `"Qwen/Qwen2.5-7B-Instruct"`, `budget="2.5GiB"`).

### 기존 스크립트를 상한 안에서 실행

**사전 요구 사항**: `pip install memopro numpy`. macOS 또는 Linux가 필요합니다.

아래 스크립트는 보통 약 1.8GB를 씁니다. 파일로 저장한 뒤 memopro로 실행하면 1GB 안에서 같은 결과를 냅니다.

```python
# big_job.py: an ordinary NumPy program, no memopro code in it
import numpy as np

x = (np.arange(200_000_000, dtype=np.float32) % 1000) / 1000
y = np.sqrt(x) * 2
print("sum:", float(y.sum(dtype=np.float64)))
```

```bash
python big_job.py                          # 최대 약 1.8GB
memopro run --budget 1GB big_job.py        # 최대 약 1GB, 같은 결과
```

### 사용 가능한 메모리 확인

```bash
memopro doctor
```

## 고급 사용법과 벤치마크

### 학습 속도: `batch="auto"`

```python
r = memopro.finetune(path, texts, budget="2.5GiB", batch="auto")
print(r.batch)                           # 고른 batch
r.adapter.save_pretrained("my-lora")     # 표준 PEFT 어댑터
```

가중치는 한 단계에 한 번만 읽히므로 batch가 클수록 토큰당 시간이 줄어듭니다. 결과는 같은 batch끼리 예산과 관계없이 같습니다.

### 생성 속도: 초안 모델

```python
text = memopro.generate(r.model, "Summarize: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

작은 모델이 토큰을 제안하고 큰 모델이 검증합니다. 출력은 큰 모델만으로 생성한 결과와 같습니다.

### 비전 모델

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model(path_to_dinov2, budget="1GB", device="cuda",  # "mps", "cpu"
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

### 벤치마크

| 작업 | 환경 | 결과 |
|---|---|---|
| Qwen2.5-7B LoRA 학습 | Apple M1, 8GB | 예산 2.5GiB, `batch=4`에서 15.5토큰/초 |
| Mistral-7B LoRA 학습 | NVIDIA T4, 16GB | 예산 4GiB, 약 43토큰/초 |
| Qwen2.5-7B 생성 (`draft` 사용) | Apple M1, 8GB | 8.95 → 2.37초/토큰, 출력 동일 |
| Qwen2.5-3B LoRA 학습 | Apple M1, 8GB | 20토큰/초 (mlx-tune은 메모리 부족으로 실행 불가) |
| 1.5B~3.8B 모델 (Qwen2.5, SmolLM2, Phi-3.5) | NVIDIA T4, 16GB | 메모리를 충분히 쓴 실행 대비 학습 1.00~1.03배, 생성 1.06~1.35배 시간 |
| NumPy·scikit-learn·CSV 처리, 필요 메모리의 50~75% 상한 | Linux | 결과 동일, 1.1~3.4배 시간 |

검증한 모델: Qwen2.5(1.5B~14B), SmolLM2, Phi-3.5, Mistral-7B, DINOv2, ConvNeXtV2, ResNet-152. 원시 결과는 [`docs/research/data/`](docs/research/data/)에 있습니다.

### 한계

- 메모리를 크게 넘는 모델은 실행되지만 저장 장치 읽기 속도에 묶여 매우 느립니다. 예를 들어 NVIDIA T4에서 14B 16비트 LoRA는 약 1토큰/초입니다.
- 무작위 접근이 많은 데이터 작업(정렬, 조인, Polars 등)은 결과는 같지만 크게 느려질 수 있습니다.
- Apple GPU에서는 `finetune`/`stream_model`을 다른 MPS 연산보다 먼저 호출해야 메모리 설정이 적용됩니다.
- 알파 버전이라 API가 바뀔 수 있습니다. 변경 내역은 [CHANGELOG](CHANGELOG.md)에 있습니다.

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
