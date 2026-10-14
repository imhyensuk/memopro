# memopro

**메모리가 부족해서 돌아가지 않던 작업을, 결과는 그대로 둔 채 돌아가게 합니다.**

[![PyPI](https://img.shields.io/pypi/v/memopro.svg)](https://pypi.org/project/memopro/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#라이선스)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

한국어 · [English](README.en.md)

```bash
pip install "memopro[llm]"
```

```python
import memopro

# 8GB 맥북에서 7B 모델(가중치 14GB)을 16비트 그대로 LoRA 학습
r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="2GiB")
print(memopro.generate(r.model, "안녕하세요"))
```

---

## 이럴 때 쓰세요

- 🧠 **내 노트북보다 큰 언어 모델을 학습하거나 쓰고 싶을 때**: 8GB 맥북에서 7B 모델을 학습하고, 무료 Colab GPU에서 GPU보다 큰 모델을 돌립니다.
- 🖼️ **큰 비전 모델을 작은 메모리에서 추론할 때**: 받아 둔 Hugging Face 모델을 메모리보다 커도 그대로 씁니다.
- 📊 **데이터가 RAM보다 커서 스크립트가 "메모리 부족"으로 꺼질 때**: 코드를 고치지 않고 `memopro run` 한 줄로 실행합니다.
- 🎯 **결과가 원래와 똑같아야 할 때**: 4비트 양자화처럼 정확도를 깎지 않습니다. 출력과 손실이 비트 단위로 같습니다.

memopro는 세 가지를 약속합니다.

- ✅ **결과가 바뀌지 않습니다.** 메모리를 아끼는 대신 시간을 씁니다.
- ✅ **정한 메모리 상한 안에서 돕니다.** 안 되는 예산이면 시작 전에 알려 주고, 코드를 고치지 않은 프로그램이 상한을 넘을 수밖에 없으면 그 순간 경고하고 얼마나 넘었는지 보고합니다.
- ✅ **디스크에 쓰지 않습니다.** 스왑 파일이나 캐시 파일을 만들지 않습니다.

---

## 사용법

### 1. 맥북에서 언어 모델 미세조정

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-7B-Instruct", texts,
    budget="2.5GiB",   # 쓸 메모리 상한
    batch="auto",      # 상한에 들어가는 가장 큰 batch를 알아서 고름 (클수록 빠름)
)
r.adapter.save_pretrained("my-lora")   # 표준 PEFT 어댑터로 저장
print(r.losses, r.batch)
```

> 8GB M1에서 7B 학습이 초당 약 15토큰입니다. 같은 데이터를 다른 예산으로 학습해도 손실이 비트 단위로 같습니다.

### 2. 글 생성 (빠르게, 결과는 그대로)

```python
text = memopro.generate(r.model, "요약해 줘: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

`draft`로 작은 모델을 주면 작은 모델이 먼저 답을 추측하고 큰 모델은 확인만 합니다. 출력은 큰 모델 혼자 만든 것과 똑같고, 7B에서 3.8배 빨라집니다.

### 3. NVIDIA GPU(Colab 등)에서 GPU보다 큰 모델

```python
# 코드는 맥과 같습니다. NVIDIA GPU가 있으면 알아서 씁니다
r = memopro.finetune("mistralai/Mistral-7B-Instruct-v0.3", texts, budget="4GiB", batch="auto")
```

> 무료 Colab T4(15GB)에서 7B 16비트 LoRA를 한 단계(256토큰)에 약 6초로 학습합니다.

### 4. 큰 비전 모델 추론

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).pooler_output   # 그냥 불러온 모델과 같은 출력
```

### 5. 쓰던 스크립트를 그대로 (코드 수정 없음)

```bash
memopro run --budget 8GB analysis.py   # 큰 NumPy 배열을 상한 안에서 무손실 압축
memopro run --budget 50% train.py      # 측정한 여유 메모리의 절반 안에서
memopro run --dry-run train.py         # 무엇을 할지만 보여 줌
```

### 6. 내 코드에 한 줄 추가

```python
import memopro

s = memopro.enable(budget="8GB")                    # 이 프로세스 전체의 메모리 상한
print(s.estimate(sample, total="12GB", passes=3))   # 실행 전: 몇 초 더 걸릴지 예측
...                                                 # 평소처럼 쓴 NumPy / PyTorch 코드
print(s.measured())                                 # 실행 후: 실제 시간과 최대 메모리
```

### 7. 실행 전에 확인

```bash
memopro doctor                                                      # 장치별로 쓸 수 있는 메모리
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # 이 모델이 들어가는지
```

더 많은 예시: [사용 가이드](docs/guide/README.md)

### 예산 쓰는 법

예산은 모든 함수, `--budget`, `memopro.toml`, 환경 변수 `MEMOPRO_BUDGET`에서 같은 형식입니다.

| 형식 | 뜻 |
|---|---|
| `"auto"` | 측정한 여유 메모리 − 10% (기본) |
| `"6GB"`, `"50%"`, `0.5` | 크기, 또는 여유 메모리의 비율 |
| `"-2GB"` | 2GB는 남겨 두고 나머지 |
| `"2GB..6GB"` | 최대 6GB, 2GB도 안 되면 시작하지 않음 |
| `"6GB!"` | 측정과 관계없이 6GB (스왑 위험 감수) |

---

## 실험 성과

모든 결과는 실행 전에 판정 기준을 정해 두고 쟀습니다. 원시 데이터는 [`docs/research/data/`](docs/research/data/)에 있습니다.

### 🍎 MacBook Air M1 (메모리 8GB)

| 해 본 것 | 결과 |
|---|---|
| Qwen2.5-**7B** LoRA 학습 (가중치 14GB) | ✅ 예산 1.5GiB로 완주. `batch=4`로 초당 **15.5토큰**. 예산이 달라도 손실 동일 |
| Qwen2.5-3B LoRA, mlx-tune과 비교 | ✅ mlx-tune은 메모리 부족으로 시작 못 함, memopro는 1GiB로 초당 20토큰 |
| Qwen2.5-7B 생성 (초안 모델 사용) | ✅ 출력 동일, 8.95 → **2.37초/토큰** |
| Qwen2.5-3B 생성, llama.cpp와 비교 | ✅ memopro 1.09초/토큰, llama.cpp(CPU) 14.75초/토큰 |
| NumPy 영상 처리, 메모리 절반 상한 | ✅ 결과 동일, 상한 +1.8MiB 이내. 느려짐 예측 3.81초, 실제 3.72초 |

### 🟩 무료 Google Colab T4 (GPU 15GB, RAM 12.7GB)

PyPI 버전을 새 런타임에 설치해 시험했습니다(0.1.4).

| 해 본 것 | 결과 |
|---|---|
| 언어 모델 5종: Qwen2.5-1.5B, SmolLM2-1.7B, Phi-3.5-mini, Mistral-7B, Qwen2.5-14B | ✅ 생성 토큰과 학습 손실이 예산과 관계없이 같음 |
| 작은 모델(1.5B~3.8B)의 속도 | 학습 **1.00~1.03배**, 생성 1.06~1.35배 (메모리를 덜 쓰고도 거의 같은 속도) |
| Mistral-**7B** 16비트 LoRA | ✅ 한 단계(256토큰) 약 6초, 초당 약 43토큰 |
| Qwen2.5-**14B** 16비트 LoRA | ⚠️ 완주했지만 단계당 약 125초(초당 약 1토큰). 일반 실행과 Unsloth 16비트는 실패. **가능하지만 실용적이지는 않습니다** |
| 비전: DINOv2-giant, ConvNeXtV2-Huge, ResNet-152 | ✅ 출력이 일반 실행과 비트 단위로 같음 |
| NumPy·scikit-learn·CSV·이미지 전처리, 메모리 절반·¾ 상한 | ✅ 결과 동일. 상한을 지키거나, 못 지킬 때는 경고·보고. 1.1~3.4배 느려짐 |

### 어디까지 실용적인가요

| 상황 | 기대할 수 있는 것 |
|---|---|
| 메모리를 **조금** 넘는 모델 (7B on 8GB 맥, 7B on T4) | ✅ 실용적. 초당 8~45토큰 |
| 메모리에 **다 들어가는** 모델 | 쓸 이유가 적습니다. Unsloth·mlx-tune이 더 빠릅니다(T4에서 10~33배) |
| 메모리를 **크게** 넘는 모델 (14B on T4) | ⚠️ 돌아가지만 저장 장치 읽기 속도에 묶여 매우 느립니다 |
| 정확도를 조금 잃어도 된다면 | 4비트 QLoRA가 훨씬 빠릅니다 |
| 무작위 접근이 많은 데이터 작업 (정렬·조인·Polars) | ⚠️ 결과는 맞지만 많이 느려질 수 있습니다 |

---

## 설치와 지원 환경

| 설치 | 용도 |
|---|---|
| `pip install memopro` | `memopro run`, `enable`, 런타임, 진단 |
| `pip install "memopro[llm]"` | `finetune`, `generate` (PyTorch·Transformers·PEFT 포함) |
| `pip install "memopro[notebook]"` | Jupyter 매직 (`%hibernate`, `%wake`) |

| 환경 | 상태 |
|---|---|
| 🍎 macOS, Apple silicon | 주 개발 환경. 모든 기능 |
| 🟩 Linux + NVIDIA GPU | Colab T4에서 검증 |
| 🐧 Linux CPU | CI에서 검증 |
| 🪟 Windows | 기본 기능만 |

Python 3.11 이상, PyTorch 2.4 이상. PyPI에 macOS arm64, Linux x86_64·aarch64 wheel이 있습니다. Rust([crates.io](https://crates.io/crates/memopro))와 C([`crates/memopro-c`](crates/memopro-c))에서도 쓸 수 있습니다.

---

## 어떻게 가능한가요

지금 쓰지 않는 데이터를 메모리에서 비웠다가 필요할 때 되돌립니다. 언어 모델이라면 계산할 층의 가중치만 원본 파일에서 읽고 다 쓰면 내립니다. 그 밖의 데이터는 바이트 그대로 되돌릴 수 있는 무손실 압축을 씁니다. 근사나 양자화를 하지 않으므로 결과는 바뀌지 않고 시간만 더 듭니다.

## 알아 두면 좋은 점

- Apple GPU에서는 `finetune`/`stream_model`을 다른 MPS 작업보다 먼저 부르세요. 그래야 메모리 설정이 적용됩니다(늦으면 경고가 뜹니다).
- CLIP·SigLIP처럼 텍스트와 비전이 한 파일에 든 모델의 비전 부분만 쓰기는 다음 버전에서 고쳐집니다.
- 알파 버전이라 API가 바뀔 수 있습니다. 변경 내역은 [CHANGELOG](CHANGELOG.md)에 있습니다.

## 개발

```bash
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest -q
cargo test -p memopro
```

## 인용

연구에 사용하셨다면 저장소의 **"Cite this repository"**([`CITATION.cff`](CITATION.cff))로 인용해 주세요.

## 라이선스

[MIT](LICENSE-MIT) 또는 [Apache-2.0](LICENSE-APACHE) 중 선택.
