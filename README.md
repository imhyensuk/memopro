# memopro

**메모리가 부족해서 돌아가지 않던 작업을, 결과는 그대로 둔 채 돌아가게 합니다.**

[![PyPI](https://img.shields.io/pypi/v/memopro.svg)](https://pypi.org/project/memopro/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#라이선스)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

한국어 · [English](README.en.md)

8GB 노트북에 14GB짜리 언어 모델을 올려 학습하고 싶을 때, 데이터가 RAM보다 커서 프로그램이 "메모리 부족"으로 꺼질 때 쓰는 오픈 소스 라이브러리입니다.

```bash
pip install "memopro[llm]"
```

```python
import memopro

# 8GB Mac에서 7B 모델(가중치 14GB)을 16비트 LoRA로 학습 — 메모리는 1.5GiB만 씁니다
r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="1.5GiB")
print(memopro.generate(r.model, "안녕하세요"))
```

---

## ✨ memopro로 할 수 있는 것

| | 기능 | 한 줄 설명 |
|---|---|---|
| 🧠 | **메모리보다 큰 LLM 미세조정** | 8GB Mac에서 7B 모델을 16비트 그대로 LoRA 학습 |
| 💬 | **메모리보다 큰 LLM으로 생성** | 출력은 모델을 통째로 올렸을 때와 토큰 하나까지 같음 |
| 🖼️ | **큰 비전·언어 모델 추론** | 받아 둔 Hugging Face 모델을 메모리보다 커도 그대로 실행 |
| 🪄 | **코드 수정 없이 메모리 상한 걸기** | `memopro run --budget 8GB script.py` 한 줄로 NumPy·PyTorch 스크립트를 8GB 안에서 |
| 📦 | **큰 배열·객체를 예산 안에 보관** | 쓰지 않는 동안 무손실 압축하거나 원본에서 다시 읽음 |
| 🔮 | **얼마나 느려질지 미리 알려 줌** | 실행 전에 "이 상한이면 몇 초 더 걸린다"를 예측 |
| 🩺 | **내 기기에 들어가는지 미리 확인** | `memopro doctor`, `memopro check`로 실행 전에 진단 |

그리고 어떤 기능이든 세 가지는 지킵니다.

- ✅ **결과가 바뀌지 않습니다.** 양자화나 근사를 쓰지 않습니다. 출력과 손실이 비트 단위로 같습니다.
- ✅ **정한 메모리 상한을 넘지 않습니다.** 불가능한 예산이면 시작하기 전에 알려 줍니다.
- ✅ **디스크에 쓰지 않습니다.** 스왑 파일이나 캐시 파일을 만들지 않습니다.

> "메모리"는 GPU·RAM 같은 하드웨어 메모리입니다. 에이전트의 대화 기억과는 관계없습니다.

---

## 🚀 사용 예시

### 1. 8GB Mac에서 7B 모델 미세조정

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-7B-Instruct", texts,
    budget="2.5GiB",   # 가중치와 활성값에 쓸 메모리 상한
    seq_len=129,
    batch="auto",      # 예산에 들어가는 가장 큰 batch: 한 번 읽은 가중치를 여러 시퀀스가 나눠 씀
)
print(r.batch, r.tokens / r.seconds)    # 8GB M1, batch 4에서 약 15.5토큰/초
r.adapter.save_pretrained("my-lora")    # 표준 PEFT 어댑터로 저장
```

### 2. 미세조정한 모델로 글 생성

```python
text = memopro.generate(r.model, "요약해 줘: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

`draft`를 주면 작은 모델이 먼저 답을 추측하고, 큰 모델은 그것을 검증만 합니다. 결과는 큰 모델 혼자 만든 것과 똑같고, 속도는 여러 배 빨라집니다(7B: 8.95 → 2.37초/토큰).

### 3. 메모리보다 큰 비전 모델 추론

```python
import transformers
import memopro.rt.torch as rtt

m = rtt.stream_model("facebook/dinov2-giant", budget="1GB", device="cuda",  # "mps", "cpu"
                     model_class=transformers.Dinov2Model)
features = m(pixel_values=images).last_hidden_state   # 그냥 불러온 모델과 같은 출력
```

### 4. 쓰던 스크립트를 고치지 않고 8GB 안에서

```bash
memopro run --budget 8GB analysis.py   # 큰 NumPy 배열은 상한 안에서 무손실 압축
memopro run --budget 6GB train.py      # 안 들어가는 from_pretrained는 예산에 맞춰 불러옴
memopro run --dry-run train.py         # 무엇을 할지만 보여 줌
```

### 5. 내 코드에 한 줄 추가

```python
import memopro

s = memopro.enable(budget="8GB")                    # 이 프로세스 전체의 메모리 상한
print(s.estimate(sample, total="12GB", passes=3))   # 실행 전: 얼마나 느려질지
...                                                 # 평소처럼 작성한 NumPy / PyTorch 코드
print(s.measured())                                 # 실행 후: 실제로 쓴 시간
```

### 6. 큰 배열과 객체를 직접 맡기기

```python
from memopro.rt import Runtime

rt = Runtime(budget="2GB")
big = rt.load_npy("big.npy")                 # 원본 파일: 필요하면 버렸다가 다시 읽음(해시로 확인)
work = rt.array((50_000, 4_096), "float32")  # 새 버퍼: 쉬는 동안 무손실 압축
with work.view(write=True) as a:             # 쓰는 동안만 메모리에, 복사 없는 NumPy 뷰
    a[:] = 1.0

h = rt.adopt(state)    # 이미 가진 텐서·모듈·옵티마이저·KV 캐시·배열 dict도 맡길 수 있음
with h:
    step(state)
```

### 7. 실행 전에 진단하기

```bash
memopro doctor                                                      # 장치·RAM·디스크별 쓸 수 있는 메모리
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # 이 모델이 들어가는지, 어떻게
```

더 자세한 안내는 [사용 가이드](docs/guide/README.md)에 있습니다.

---

## 🌍 범용성

### 어디서 돌아가나요

| 환경 | 상태 |
|---|---|
| 🍎 macOS, Apple silicon (MPS) | 주 개발 환경. LLM 학습·생성, 코드 수정 없는 메모리 상한 |
| 🟩 Linux + NVIDIA GPU (CUDA) | Colab T4에서 생성, LoRA 학습, 비전 추론 검증 |
| 🐧 Linux, CPU | CI(x86_64)에서 검증. 코드 수정 없는 메모리 상한 (aarch64는 wheel만 제공) |
| 🪟 Windows | 기본 기능만 확인 |

PyPI에 macOS arm64, Linux x86_64·aarch64용 wheel이 있습니다. 그 밖의 환경은 Rust 툴체인으로 소스에서 빌드합니다. Python 3.11 이상, PyTorch 2.4 이상이 필요합니다.

### 무엇을 다룰 수 있나요

| 분야 | 확인한 예 |
|---|---|
| 언어 모델 학습·생성 | Qwen2.5 1.5B·3B·7B (Hugging Face safetensors) |
| 비전 모델 추론 | ResNet-152, DINOv2-giant(11억 파라미터) |
| 일반 데이터 작업 | 수정하지 않은 NumPy 영상 처리, scikit-learn 분류, 반복 배열 계산 |
| 직접 만든 코드 | PyTorch 학습의 활성값, 텐서·모듈·옵티마이저·KV 캐시·배열을 담은 객체(`adopt`) |

설계는 메모리 크기와 무관합니다. 요구량이 보유량보다 크면 어떤 기기에서든 같은 방식으로 동작하도록 만들었고([설계](docs/design/scale.md)), 지금까지는 8~16GB 기기와 작은 GPU에서 검증했습니다.

### 실제로 재 본 결과

모두 실행 전에 기준을 정해 두고 쟀습니다. 원시 데이터는 [`docs/research/data/`](docs/research/data/), 스크립트는 [`experiments/`](experiments/)에 있습니다.

**언어 모델 (MacBook Air M1, 메모리 8GB)**

| 작업 | 결과 |
|---|---|
| Qwen2.5-**7B** LoRA 학습 (가중치 14.2GiB) | 예산 1.5GiB로 완주. 129토큰, 예산 2.5GiB: batch 1 8.4토큰/초, `batch=4` **15.5토큰/초**. 예산이 달라도 손실이 비트 단위로 같음 |
| Qwen2.5-3B LoRA, mlx-tune과 같은 설정 | mlx-tune은 메모리 부족으로 시작조차 못 함. memopro는 1GiB 예산으로 20.0토큰/초 |
| Qwen2.5-7B 생성 (초안 + 검증) | 일반 생성과 출력 동일, 8.95 → 2.37초/토큰 (3.8배) |
| Qwen2.5-7B 생성, 0.1.1 읽기 경로 | 출력 동일, 같은 기기 상태에서 0.1.0 대비 9.39 → **7.24초/토큰 (+30%)** |
| Qwen2.5-3B 생성, 다른 도구와 비교 | memopro 1.09초/토큰(2.1GB). llama.cpp CPU 14.75초/토큰(3.3GB). llama.cpp Metal은 메모리 부족 |
| Qwen2.5-7B LoRA, Colab T4(15GB) | 그냥은 GPU 메모리 부족. memopro는 예산 4GiB로 완주 |

**일반 프로그램 (원래 필요한 메모리의 절반 상한)**

| 작업 | 결과 |
|---|---|
| 수정하지 않은 NumPy 영상 처리 (Linux) | 결과 동일, 같은 상한의 OS 스왑보다 3.3배 빠름 |
| 수정하지 않은 scikit-learn 분류 (Linux) | 결과 동일. 그대로 실행하면 메모리 부족으로 종료 |
| NumPy 영상 처리, `memopro run` (M1) | 결과 동일, 최대 메모리가 상한 +1.8MiB 이내 |
| 느려짐 예측 (M1) | 예측한 추가 시간 3.81초, 실제 3.72초 |
| 메모리가 넉넉할 때 켜 둔 비용 | macOS 1.02~1.08배, Linux 1.20배 |

**비전 모델 (Colab T4)**

| 모델 | 결과 |
|---|---|
| ResNet-152 | 예산 60MiB에서도 출력이 비트 단위로 같음, 0.13 → 0.14초 |
| DINOv2-giant (4.5GB) | 예산 1.1GB에서 출력 동일, 1.57 → 1.65초 |

### 이럴 때는 다른 도구가 낫습니다

- 모델이 메모리에 **다 들어가면** Unsloth, mlx-tune, 일반 PyTorch가 더 빠릅니다. Colab T4에서 Unsloth는 3B·7B LoRA를 10~33배 빠르게 학습했습니다.
- 정확도를 **조금 잃어도 된다면** 4비트 양자화(QLoRA, llama.cpp)가 빠릅니다.
- memopro는 그 도구들로는 **돌아가지 않는 크기**, 그리고 **결과가 원래와 같아야 하는** 경우를 위한 것입니다.

---

## 🔌 호출 방식

memopro는 필요한 만큼만 골라 쓸 수 있도록 네 가지 방식으로 부를 수 있습니다.

### 1. 한 줄 API (Python)

| 함수 | 하는 일 |
|---|---|
| `memopro.finetune(model, texts, budget=, batch=)` | 메모리보다 큰 LLM의 LoRA 학습 (`batch="auto"`: 예산에 맞는 가장 큰 batch) |
| `memopro.generate(model, prompt, draft=)` | 메모리보다 큰 LLM의 생성(출력은 일반 생성과 같음) |
| `memopro.enable(budget=)` / `disable()` | 이 프로세스 전체에 메모리 상한(NumPy·PyTorch 자동 처리) |
| `memopro.load(name, quality=)` | 예산에 맞는 형식(필요하면 양자화)을 골라 모델 불러오기 |
| `memopro.train_session(model, optimizer)` | 예산에 맞게 마이크로배치·체크포인팅을 골라 정확한 학습 |
| `memopro.doctor()` / `memopro.check(...)` | 쓸 수 있는 메모리 진단, 모델이 들어가는지 예측 |

### 2. 런타임 API (직접 다루고 싶을 때)

| 이름 | 하는 일 |
|---|---|
| `memopro.rt.Runtime(budget=)` | 예산 안에서 버퍼를 관리하는 런타임 |
| `rt.load_npy`, `rt.array`, `rt.add_file` | 파일·새 버퍼를 등록하고 NumPy 뷰로 사용 |
| `rt.adopt(obj)` | 이미 가진 객체를 런타임에 맡김 |
| `memopro.rt.torch.stream_model(name, budget=, device=)` | Hugging Face 모델을 가중치는 파일에 둔 채로 실행 |

### 3. 명령줄

| 명령 | 하는 일 |
|---|---|
| `memopro run --budget 8GB script.py` | 스크립트를 고치지 않고 메모리 상한 안에서 실행 |
| `memopro doctor` | 장치·RAM·디스크별 쓸 수 있는 메모리 |
| `memopro check MODEL --batch-size 4 --seq-len 512` | 모델이 예산에 들어가는지, 어떤 설정이면 되는지 |

### 4. 노트북과 다른 언어

- **Jupyter**: `%load_ext memopro` 뒤에 `%hibernate 변수`(잠시 메모리에서 내림), `%wake 변수`, `%memopro status`
- **Rust**: crates.io [`memopro`](https://crates.io/crates/memopro)
- **C/C++**: C ABI [`crates/memopro-c`](crates/memopro-c)

### 설치 옵션과 예산 쓰는 법

| 설치 | 용도 |
|---|---|
| `pip install memopro` | 런타임, `enable`, `memopro run`, 진단 |
| `pip install "memopro[llm]"` | `finetune`, `generate` (PyTorch·Transformers·PEFT 포함) |
| `pip install "memopro[torch]"` / `"memopro[hf]"` | PyTorch 연동 / Hugging Face 모델 불러오기 |
| `pip install "memopro[notebook]"` | Jupyter 매직 |

예산은 모든 API, CLI의 `--budget`, `memopro.toml`, 환경 변수 `MEMOPRO_BUDGET`에서 같은 형식으로 씁니다.

| 형식 | 의미 |
|---|---|
| `"auto"` | 측정한 여유 메모리 − 10% (기본) |
| `"6GB"`, `0.5`, `"50%"` | 크기, 또는 측정값의 비율 |
| `"-2GB"` | 측정값에서 2GB는 남겨 둠 |
| `"2GB..6GB"` | 최대 6GB, 2GB도 안 되면 시작하지 않음 |
| `"6GB!"` | 측정값과 관계없이 6GB (스왑 위험 감수) |
| `{"device": "80%", "host": "-2GB"}` | 메모리 종류마다 따로 |

---

## 🧩 코드 소개

```
memopro/
├── crates/
│   ├── memopro/          Rust 코어: 런타임, 페이저, 압축, 원본 다시 읽기 (crates.io)
│   ├── memopro-py/       Python 바인딩 (PyO3)
│   ├── memopro-c/        C ABI: C/C++/다른 언어에서 런타임 사용
│   └── memopro-preload/  Linux LD_PRELOAD: 큰 할당을 페이저로 보냄
├── python/memopro/
│   ├── llm.py            finetune, generate
│   ├── _enable.py        enable/disable: 프로세스 전체 상한
│   ├── _run.py, cli.py   memopro run / doctor / check
│   ├── rt/               Runtime, adopt, stream_model(가중치 흘려 쓰기), 활성값 이동
│   ├── access/           load, check, train_session, optimize (기존 기법 선택·적용)
│   ├── orchestrator/     예산 계산, 후보 구성 고르기
│   ├── hibernate/        객체를 잠시 메모리에서 내리기(동면)
│   ├── census/           학습 상태의 메모리 구성 분석
│   └── integrations/     Jupyter, Hugging Face, Lightning 연동
├── experiments/          모든 실험 스크립트 (결과 재현용)
├── docs/design/          설계 문서
└── docs/research/data/   실험 원시 결과와 실행 환경
```

- **Rust 코어**가 메모리를 실제로 관리하고, **Python**은 사용하기 쉬운 입구입니다.
- 양자화·오프로드·체크포인팅 같은 기존 기법은 다시 만들지 않고 연결해서 씁니다.
- 핵심 파일
  - [`crates/memopro/src/rt/mod.rs`](crates/memopro/src/rt/mod.rs): 버퍼마다 무엇을 할지 고르는 런타임
  - [`crates/memopro/src/rt/pager`](crates/memopro/src/rt/pager): 코드 수정 없는 페이징
  - [`python/memopro/rt/torch.py`](python/memopro/rt/torch.py): 가중치 흘려 쓰기와 일반 생성과 같은 결과를 내는 검증

---

## ⚙️ 작동 방식과 원리

### 핵심 아이디어: 메모리 대신 시간을 쓴다

memopro는 지금 쓰지 않는 데이터를 메모리에서 비우고, 필요할 때 되돌립니다. 비우는 방법은 데이터마다 다르며, **되돌리는 비용이 가장 싼 것**을 실측으로 고릅니다.

| 방법 | 언제 | 되돌릴 때 |
|---|---|---|
| 그대로 둠 | 곧 다시 쓸 데이터 | 비용 없음 |
| 무손실 압축 | 메모리에서 만든 데이터 | 압축 풀기 (바이트 그대로) |
| 원본 다시 읽기 | 파일에서 온 데이터 (모델 가중치 등) | 파일에서 읽고 해시로 같은지 확인 |
| 다시 계산 | 다른 버퍼에서 계산된 데이터 | 같은 계산을 다시 (해시로 확인) |

그래서 결과는 바뀌지 않고, 바뀌는 것은 시간뿐입니다. 비교하면 이렇습니다.

| | 운영체제 스왑 | 양자화 (4비트 등) | memopro |
|---|---|---|---|
| 결과 | 같음 | **달라짐** | **같음** (비트 단위) |
| 메모리 사용량 | 정할 수 없음 | 줄어듦 | **정한 상한을 지킴** |
| 속도 | 심하면 멈춘 듯 | 빠름 | 측정한 작업 대부분에서 스왑보다 2~5배 빠름, **미리 예측** |
| 디스크 쓰기 | 많음 | 없음 | 없음 |

### 계층 구조

```
Python API      finetune · generate · load · train_session · run · adopt
                ─────────────────────────────────────────────────────────
접근 계층       환경 감지 → 예산 → 후보 구성 선택 → 적용 → 실측 보고
                ─────────────────────────────────────────────────────────
Rust 런타임     버퍼마다 실측 비용으로 선택: 유지 · 무손실 압축 · 원본 다시 읽기 · 재계산
                + 미리 읽기, 메모리 상한 보장, 느려짐 예측
                ─────────────────────────────────────────────────────────
플랫폼          Apple GPU 무복사 버퍼 · CUDA 비동기 복사 · Linux userfaultfd · macOS 신호 기반 페이저
```

### 메모리보다 큰 LLM은 어떻게 도나요

1. **가중치는 파일에 둡니다.** 층을 계산할 차례가 되면 그 층의 가중치만 원본 safetensors에서 읽어 오고, 다 쓰면 내립니다. 다음 층은 미리 읽어 계산과 겹칩니다.
2. **복사하지 않습니다.** Apple GPU에는 읽어 온 메모리를 그대로 넘깁니다. CUDA에서는 다음 층을 옆 스트림으로 미리 복사합니다.
3. **SSD를 최대한 빠르게 읽습니다.** 페이지 경계에 맞춰 여러 스레드로 읽고, OS 캐시에 남기지 않습니다(0.1.1에서 생성 +30%).
4. **학습은 활성값까지 예산에 넣습니다.** 층별 체크포인팅을 쓰고, 활성값 크기를 미리 추정했다가 실측으로 고칩니다. 한 스텝을 담을 수 없는 예산은 시작 전에 거절합니다.
5. **생성 결과를 바꾸지 않고 빠르게 합니다.** 작은 초안 모델이 토큰을 추측하면, 큰 모델이 한 번 가중치를 읽을 때 여러 토큰을 함께 검증합니다. 검증 행을 하나씩 일반 생성과 같은 모양으로 계산하므로 출력이 토큰 하나까지 같습니다.

### 코드 수정 없는 메모리 상한은 어떻게 하나요

- 큰 NumPy 배열을 런타임이 관리하는 메모리에 둡니다.
- 프로그램이 내려간 부분에 접근하면 페이저가 그 순간 압축을 풀어 되돌립니다. Linux는 `userfaultfd`를, macOS는 접근 신호를 씁니다.
- 프로세스 전체 메모리를 계속 재서, 상한을 넘지 않도록 다른 부분을 압축합니다.
- 데이터 표본으로 압축률과 속도를 재서 "이 상한이면 몇 초 더 걸린다"를 예측합니다(같은 순서로 반복해서 읽는 작업 기준).

설계 문서: [아키텍처](docs/design/architecture.md) · [런타임](docs/design/runtime.md) · [LLM 경로](docs/design/g4-architecture.md) · [규모 무관 설계](docs/design/scale.md)

---

## 한계

- **속도**: 메모리를 아끼는 대신 시간이 듭니다. 7B 학습은 8GB Mac에서 129토큰 스텝당 약 15초입니다(`batch=4`면 시퀀스 4개에 약 33초). 메모리에 다 들어가는 모델은 기존 도구가 더 빠릅니다.
- **코드 수정 없는 페이징**: 무작위 접근이 많은 작업(정렬·그룹 집계)과 압축해도 상한보다 큰 데이터에서는 실용적인 속도가 나지 않습니다. Linux와 macOS에서 동작합니다(Windows 미지원).
- **모델 범위**: LLM은 주로 Qwen2.5 계열(1.5B~7B), 비전은 ResNet-152와 DINOv2로 검증했습니다. 확산 모델·음성 모델은 시험하지 않았습니다.
- **Apple GPU**: `finetune`/`stream_model`은 MPS를 쓰기 전에 부르세요. 그래야 torch MPS 할당기 설정이 적용됩니다(이미 쓰고 있으면 경고가 뜹니다).
- **알파 버전**: API가 바뀔 수 있습니다. 변경 내역은 [CHANGELOG](CHANGELOG.md)에 있습니다.

---

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
