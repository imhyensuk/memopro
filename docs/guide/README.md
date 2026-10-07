# memopro 사용 안내

메모리가 부족한 기기에서 원래 메모리보다 큰 작업을 **결과를 바꾸지 않고**, **정한 상한 안에서** 돌리는 방법입니다. 영어판은 [README.en.md](README.en.md)입니다.

- [1. 설치](#1-설치)
- [2. 한 줄로 시작하기](#2-한-줄로-시작하기)
- [3. 코드 수정 없이 스크립트 실행](#3-코드-수정-없이-스크립트-실행)
- [4. 실행 전에 얼마나 느려질지 보기](#4-실행-전에-얼마나-느려질지-보기)
- [5. NumPy 데이터 처리](#5-numpy-데이터-처리)
- [6. PyTorch 학습](#6-pytorch-학습)
- [7. 메모리보다 큰 LLM](#7-메모리보다-큰-llm)
- [8. 명시적으로 제어하기: `memopro.rt`](#8-명시적으로-제어하기-memoprort)
- [9. 동작 방식과 한계](#9-동작-방식과-한계)

---

## 1. 설치

```bash
pip install memopro            # 핵심 (의존성 없음)
pip install "memopro[numpy]"   # NumPy 작업
pip install "memopro[llm]"     # LLM 학습·생성 (torch, transformers, peft)
```

Python 3.11 이상, macOS(Apple silicon) 또는 Linux가 필요합니다. Windows에서는 기본 기능만 동작합니다.

## 2. 한 줄로 시작하기

```python
import memopro

s = memopro.enable()             # 자동: (지금 쓰는 양 + 여유 메모리) - 10%
# s = memopro.enable("8GB")      # 직접: 이 프로세스 전체를 8GB 안에서
print(s)
```

`print(s)`를 하면 이 기기에서 무엇을 켰는지 보입니다.

```
memopro.enable: Darwin arm64, 8.00 GiB memory, no GPU
  ceiling       1.40 GiB for the whole process (1.40 GiB)
  process       20.12 MiB in use at start
  rt            applied: runtimes and pagers share the ceiling; ...
  numpy         applied: arrays >= 16 MiB are paged within the ceiling (compressed)
  torch         applied: activations saved for backward move ... (this thread, when torch is imported)
```

- **상한은 프로세스 전체 기준입니다.** macOS에서는 활동 모니터에 보이는 메모리, Linux에서는 RSS를 기준으로 합니다.
- 상한 형식: `"8GB"`(최대), `0.5`(측정값의 절반), `"-2GB"`(2GB는 남김), `"2GB..8GB"`(2GB 미만이면 중단), `"8GB!"`(측정값보다 커도 그대로).
- 끄려면 `memopro.disable()`을 부르거나 `with memopro.enable("8GB"):` 블록을 쓰면 됩니다.
- macOS에서는 Python을 `MallocLargeCache=0`으로 시작하는 것이 좋습니다. 해제된 메모리가 OS로 바로 돌아가기 때문입니다. `memopro run`은 이 설정을 자동으로 합니다.

## 3. 코드 수정 없이 스크립트 실행

```bash
memopro run --budget 4GB analysis.py arg1 arg2
memopro run --budget 4GB --report-json report.json train.py
memopro run --dry-run analysis.py      # 무엇을 할지만 출력
```

스크립트가 `memopro.enable("4GB")` 세션 안에서 돌아갑니다. 끝나면 무엇이 얼마나 압축되고 돌아왔는지 stderr에 출력합니다. `--no-numpy`, `--no-torch`로 각 부품을 끌 수 있습니다.

## 4. 실행 전에 얼마나 느려질지 보기

```python
import numpy as np, memopro

s = memopro.enable("2GB")
sample = np.load("part-000.npy")           # 실제 데이터의 일부
print(s.estimate(sample, total="6GB", passes=3, compute_seconds=12.0))
# {'fits': True, 'ratio': 0.34, 'moved_per_pass': ..., 'extra_seconds': 9.8, 'slowdown': 1.27, ...}
```

- 샘플을 이 기기에서 실제로 압축해 보고 압축률과 속도를 잰 뒤, 상한에서 패스마다 얼마나 압축하고 되돌려야 하는지 계산합니다.
- `fits: False`이면 압축해도 상한에 들어가지 않는다는 뜻입니다(디스크를 쓰지 않으므로 이 상한으로는 실행할 수 없습니다).
- 정해진 순서로 반복 접근하는 작업을 가정합니다. 8GB M1에서 잰 오차는 7% 이내였습니다(E047). 무작위 접근 작업은 이보다 느립니다.
- 실행한 뒤에는 `s.measured()`가 memopro가 실제로 쓴 시간과 배율을 알려 줍니다.

## 5. NumPy 데이터 처리

`enable()` 뒤에 만든 16 MiB 이상의 배열은 자동으로 상한 안에서 관리됩니다. 코드는 그대로입니다.

```python
import numpy as np, memopro

with memopro.enable("1GB"):
    stack = np.empty((4096, 512, 512), np.uint16)   # 2 GiB: 상한보다 큼
    for i in range(len(stack)):
        stack[i] = load_image(i)
    means = stack.mean(axis=(1, 2))
    np.save("stack.npy", stack)                     # 그대로 동작
```

- 넘치는 부분은 메모리 안에서 무손실 압축되고, 필요할 때 비트 그대로 돌아옵니다. 디스크에는 쓰지 않습니다.
- `np.save`, `np.load`, `np.fromfile`은 자동으로 처리됩니다. `ndarray.tofile`이나 `file.readinto(배열)`처럼 커널이 페이징된 배열을 직접 읽고 쓰는 경우는 `OSError`가 날 수 있습니다. 데이터가 틀리게 나오는 일은 없습니다.
- 메모리가 넉넉할 때 켜 둔 비용은 1.02~1.08배였습니다(E047).

## 6. PyTorch 학습

```python
import memopro
s = memopro.enable("6GB")      # torch를 import하기 전에 부르는 것이 좋습니다
import torch
...                            # 평소의 학습 루프
```

- 프로세스가 상한의 75%를 넘으면, 역전파를 위해 저장된 활성값(1 MiB 이상)을 memopro 버퍼로 옮깁니다. 바이트가 그대로 돌아오므로 기울기는 바뀌지 않습니다(CPU, Apple GPU 확인).
- `enable()`을 부른 스레드에서만 동작합니다.
- 모델 가중치는 옮기지 않습니다. 메모리보다 큰 Hugging Face 모델은 7절의 방법을 쓰세요.

## 7. 메모리보다 큰 LLM

```python
import memopro

r = memopro.finetune("Qwen/Qwen2.5-3B-Instruct", texts, budget="1GiB", seq_len=512)
print(r.losses)
text = memopro.generate(r.model, "요약해 줘: ...", draft="Qwen/Qwen2.5-1.5B-Instruct")
```

- 16비트 가중치를 원본 safetensors 파일에서 층 단위로 흘려 씁니다. 예산이 달라도 학습 손실이 비트 단위로 같습니다.
- `enable()` 세션 안에서는 `budget="auto"`(기본값)가 세션 상한을 뜻합니다.
- `from_pretrained`로 직접 불러오는 코드도 `enable()`이나 `memopro run` 안에서는, 모델이 그대로 들어가지 않을 때만 memopro가 맞춰서 불러옵니다.

## 8. 명시적으로 제어하기: `memopro.rt`

```python
import memopro.rt as rt

r = rt.Runtime(budget="2GB")
blocks = [r.load_npy(f"part-{i:03d}.npy") for i in range(40)]   # 읽지 않고 등록만
for _ in range(3):
    for b in blocks:
        with b.view() as x:          # NumPy 배열, 복사 없음
            total += x.sum()
print(r.report())
```

- 파일에서 등록한 버퍼는 필요할 때 원본 파일에서 다시 읽습니다(해시로 검증). 메모리에서 만든 버퍼는 압축합니다. 계산으로 만든 버퍼(`derive`)는 다시 계산합니다.
- `r.adopt(obj)`는 이미 가진 텐서, 배열, 모델, KV 캐시를 런타임에 맡깁니다. `with handle:` 블록 안에서만 깨어 있습니다.
- Windows와 다른 OS에서도 동작합니다(투명 페이징 없이 명시적으로 고정해서 씀).

## 9. 동작 방식과 한계

- **무손실**: 압축(byte shuffle + zstd), 원본 파일 재읽기(해시 검증), 재계산(첫 결과와 대조)만 씁니다. 결과는 비트 단위로 같습니다.
- **디스크 쓰기 없음**: 넘치는 데이터는 메모리 안에서 압축할 뿐 스왑 파일을 만들지 않습니다. 그래서 **압축해도 상한보다 큰 데이터는 처리할 수 없습니다**(`estimate`의 `fits`).
- **옮길 수 없는 메모리**: Python, 라이브러리, 작은 객체, 직접 불러온 모델 가중치, C 확장의 내부 할당도 상한에 포함됩니다. 이것만으로 상한을 넘으면 페이저는 초과를 기록하고(`overruns`) 런타임은 `BudgetExceeded`를 냅니다.
- **무작위 접근**: 정렬이나 그룹 집계처럼 여기저기 읽는 작업은 여유가 적으면 크게 느려집니다.
- **알파 버전**: API가 바뀔 수 있습니다.
- 모든 결과와 실험은 [docs/research/data](../research/data/)와 [experiments](../../experiments/)에 있습니다.
