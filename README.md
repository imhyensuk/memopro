# memopro

**기기 메모리보다 큰 작업을, 결과를 바꾸지 않고, 정해 둔 메모리 상한 안에서 실행합니다.**

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#라이선스)
![Python ≥ 3.11](https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

한국어 · [English](README.en.md)

memopro는 8~16GB 노트북·Mac이나 작은 GPU처럼 메모리가 부족한 기기를 위한 오픈 소스 라이브러리입니다. 코어는 Rust로, 인터페이스는 Python으로 작성했습니다.

- **무손실**: 양자화나 근사를 쓰지 않습니다. 생성·추론 출력은 일반 실행과 비트 단위로 같고, 학습 손실은 예산을 바꿔도 비트 단위로 같습니다.
- **메모리 상한 보장**: 사용자가 정한 예산을 넘지 않습니다. 감당할 수 없는 예산이면 실행 전에 거절합니다.
- **디스크 쓰기 없음**: 모델 가중치는 원본 파일에서 다시 읽고, 메모리에서 만든 데이터는 무손실 압축합니다. 스왑 파일이나 캐시 파일을 만들지 않습니다.

```python
import memopro

r = memopro.finetune("Qwen/Qwen2.5-7B-Instruct", texts, budget="1.5GiB")  # 16비트 LoRA, 8GB Mac에서
r.adapter.save_pretrained("my-lora")                                       # 표준 PEFT 어댑터
print(memopro.generate(r.model, "안녕하세요", draft="Qwen/Qwen2.5-1.5B-Instruct"))
```

> "메모리"는 GPU·RAM 같은 하드웨어 메모리를 뜻합니다. 에이전트의 대화 기억과는 관계없습니다.

---

## 주요 결과

모두 사전에 정한 기준으로 측정했으며, 원시 데이터와 실행 환경은 [`docs/research/data/`](docs/research/data/), 실험 스크립트는 [`experiments/`](experiments/)에 있습니다.

### 언어 모델 (MacBook Air M1, 메모리 8GB)

| 작업 | 결과 |
|---|---|
| Qwen2.5-**7B** bf16 LoRA 학습 (가중치 14.2GiB) | 예산 1.5GiB로 완주, 예산이 달라도 손실이 비트 단위로 같음, 7.3 토큰/초 |
| Qwen2.5-3B bf16 LoRA 학습, mlx-tune과 같은 설정 | mlx-tune은 첫 스텝 전에 메모리 부족으로 실패. memopro는 1GiB 예산으로 20.0 토큰/초 |
| Qwen2.5-7B 무손실 생성 (int4 초안 + 행 불변 검증) | 일반 생성과 출력 동일, 8.95 → 2.37초/토큰 (3.8배) |
| Qwen2.5-3B bf16 추론 (CPU), 필요 메모리의 1/4 예산 | 출력 동일, OS 페이징보다 약 5배 빠름 |
| Qwen2.5-7B bf16 LoRA 학습, Colab T4(15GB) | 그냥은 GPU 메모리 부족. memopro는 예산 4GiB로 완주, 예산이 달라도 손실 동일 |
| Qwen2.5-3B bf16 생성, 다른 도구와 비교 | memopro 1.09초/토큰 (프로세스 2.1GB). llama.cpp CPU 14.75초/토큰 (3.3GB). llama.cpp Metal은 메모리 부족 |

### 일반 프로그램 (Linux, 원래 필요 메모리의 1/2 한도)

| 작업 | 결과 |
|---|---|
| 수정하지 않은 NumPy 영상 처리 | 결과 동일, 같은 한도의 OS 스왑보다 3.3배 빠름 |
| 수정하지 않은 scikit-learn 분류 | 결과 동일 (그대로 실행하면 메모리 부족으로 종료) |

무작위 접근이 많은 데이터프레임 작업과, 압축해도 한도보다 큰 데이터는 아직 1/2 한도에서 실용적인 속도로 돌지 않습니다([한계](#한계)).

---

## 설치

정식 버전은 아직 PyPI에 올리지 않았습니다(알파). 소스에서 설치하려면 Rust 툴체인이 필요합니다.

```bash
pip install "memopro[llm] @ git+https://github.com/imhyensuk/memopro"
```

| 추가 의존성 | 용도 |
|---|---|
| `memopro[torch]` | PyTorch 연동 (`load`, `train_session`, 동면) |
| `memopro[hf]` | Hugging Face 모델 불러오기 |
| `memopro[llm]` | `finetune`, `generate` (PEFT 포함) |
| `memopro[notebook]` | Jupyter 매직 |

---

## 사용법

### 1. 메모리보다 큰 LLM 학습과 생성

```python
import memopro

r = memopro.finetune(
    "Qwen/Qwen2.5-3B-Instruct", texts,
    budget="1GiB",          # 가중치에 쓸 메모리 상한
    seq_len=512, rank=8,    # LoRA 설정 (q/k/v/o)
)
print(r.losses, r.tokens / r.seconds)

text = memopro.generate(r.model, "요약해 줘: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

- 16비트 가중치를 원본 safetensors 파일에서 층 단위로 흘려 쓰며, Apple GPU에는 복사 없이 넘깁니다.
- 활성값 메모리도 예산에 포함해 계획합니다. 한 스텝을 감당할 수 없는 예산은 시작 전에 거절합니다.
- `draft`를 주면 작은 int4 초안 모델로 추측 디코딩을 합니다. 검증을 일반 생성과 같은 계산 경로로 하므로 출력이 바뀌지 않습니다.
- Apple silicon(MPS), NVIDIA CUDA(Colab T4), CPU에서 검증했습니다.

### 2. 큰 배열을 예산 안에서

```python
from memopro.rt import Runtime

rt = Runtime(budget="2GB")
weights = rt.load_npy("big.npy")             # 원본 파일: 필요하면 버렸다가 다시 읽음 (해시로 확인)
work = rt.array((50_000, 4_096), "float32")  # 새 버퍼: 필요하면 무손실 압축

with work.view(write=True) as a:             # 사용하는 동안만 메모리에, 복사 없는 NumPy 뷰
    a[:] = 1.0
print(rt.stats())
```

이미 가진 객체(텐서, 모듈, 옵티마이저, KV 캐시, NumPy 배열이 든 dict 등)도 런타임에 맡길 수 있습니다.

```python
h = rt.adopt(state)    # 쉬는 동안 런타임이 관리 (필요하면 무손실 압축)
with h:                # 블록 안에서는 평소처럼 사용
    step(state)
```

### 3. 코드 수정 없이 실행

```bash
memopro run --budget 6GB train.py          # 들어가지 않는 from_pretrained를 예산에 맞춰 불러옴
memopro run --transparent 1GB analysis.py  # Linux: 큰 NumPy 배열을 디스크 쓰기 없이 압축 페이징
memopro run --dry-run train.py             # 무엇을 할지만 출력
```

### 4. 예산에 맞춰 불러오고 학습하기

```python
model, tok = memopro.load("Qwen/Qwen2.5-7B-Instruct", tokenizer=True, quality="high")

with memopro.train_session(model, optimizer) as s:   # 마이크로배치·체크포인팅을 정확한 것부터
    for batch in loader:
        s.step(batch, lambda mb: model(**mb).loss)
```

```bash
memopro doctor                                                      # 장치·RAM·디스크별 가용 메모리와 예산
memopro check Qwen/Qwen2.5-7B-Instruct --batch-size 4 --seq-len 512 # 실행 전에 메모리 예측
```

- `quality`는 자동으로 허용할 손실의 상한입니다: `"lossless"` < `"high"` < `"balanced"`(기본) < `"low"`.
- 들어가는 구성이 없으면 `BudgetExceeded`가 실제로 불러올 수 있는 설정 목록을 함께 알려 줍니다.

### 예산 형식

모든 API와 CLI(`--budget`), `memopro.toml`, `MEMOPRO_BUDGET`에서 같은 형식을 씁니다.

| 형식 | 의미 |
|---|---|
| `"auto"` | 측정한 가용 메모리 − 여유분 10% (기본) |
| `"6GB"`, `0.5`, `"50%"` | 상한, 또는 측정값의 비율 |
| `"-2GB"` | 측정값에서 2GB를 남겨 둠 |
| `"2GB..6GB"` | 최대 6GB, 2GB를 확보하지 못하면 실행하지 않음 |
| `"6GB!"` | 측정값과 관계없이 6GB (스왑 위험 감수) |
| `{"device": "80%", "host": "-2GB"}` | 메모리 풀마다 따로 |

---

## 구조

```
Python API      finetune · generate · load · train_session · run · adopt
                ─────────────────────────────────────────────────────────
접근 계층       환경 감지 → 예산 → 후보 구성 선택 → 적용 → 실측 보고
                (양자화·오프로드·체크포인팅 등 기존 기법은 연결해서 사용)
                ─────────────────────────────────────────────────────────
Rust 런타임     버퍼마다 실측 비용으로 선택:
                유지 · 무손실 압축 · 원본 다시 읽기(해시 확인) · 재계산
                + 미리 읽기, 메모리 상한 보장, 느려짐 예측
                ─────────────────────────────────────────────────────────
플랫폼          Apple GPU 무복사 버퍼 · CUDA 비동기 복사 · Linux userfaultfd
```

| 구성 요소 | 위치 |
|---|---|
| Rust 코어 (crates.io `memopro`) | [`crates/memopro`](crates/memopro) |
| C ABI | [`crates/memopro-c`](crates/memopro-c) |
| Linux 할당 가로채기 (`LD_PRELOAD`) | [`crates/memopro-preload`](crates/memopro-preload) |
| Python 패키지 | [`python/memopro`](python/memopro) |

설계 문서: [아키텍처](docs/design/architecture.md) · [런타임](docs/design/runtime.md) · [LLM 경로](docs/design/g4-architecture.md)

---

## 지원 환경

| 환경 | 상태 |
|---|---|
| macOS, Apple silicon (MPS) | 주 개발 환경. LLM 학습·생성 검증 |
| Linux, NVIDIA GPU (CUDA) | Colab T4에서 생성, LoRA 학습, 비전 추론 검증 |
| Linux, CPU | CI에서 검증. 투명 페이징은 Linux 전용 |
| Windows | CI에서 기본 기능만 확인 |

Python 3.11 이상, PyTorch 2.4 이상.

---

## 한계

- **속도**: 메모리를 아끼는 대신 시간이 듭니다. 7B 학습은 8GB Mac에서 스텝당 약 18초(129토큰)입니다. 메모리에 다 들어가는 모델은 기존 도구가 더 빠릅니다.
- **투명 페이징**: 무작위 접근이 많은 작업(정렬·그룹 집계)과 압축해도 한도보다 큰 데이터에서는 실용적인 속도가 나지 않습니다. Linux에서만 동작합니다.
- **모델 범위**: LLM 경로는 주로 Qwen2.5 계열(1.5B~7B)로 검증했습니다.
- **알파 버전**: API가 바뀔 수 있습니다.

---

## 개발

```bash
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/pytest -q
cargo test -p memopro
```

실험 스크립트는 [`experiments/`](experiments/), 원시 결과와 실행 환경은 [`docs/research/data/`](docs/research/data/), 설계 문서는 [`docs/design/`](docs/design/)에 있습니다.

## 인용

연구에 사용하셨다면 저장소의 **"Cite this repository"**([`CITATION.cff`](CITATION.cff))로 인용해 주세요.

## 라이선스

[MIT](LICENSE-MIT) 또는 [Apache-2.0](LICENSE-APACHE) 중 선택.
