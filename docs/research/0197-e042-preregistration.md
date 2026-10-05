# 0197. E042 사전 등록: Colab T4 통합 시험 — 언어 모델·비전·데이터 작업, 결과는 Google Drive에

- **날짜**: 2026-10-05
- **유형**: experiment (사전 등록)
- **상태**: 확정. 실행 전(사용자가 Colab에서 실행)
- **관련 기록**: 0195 (환경 선택·CUDA 흘려 쓰기·비전), 0196 (E040b 개발 측정, preload), 0179 (채점표)

## 사용자 지시

> "그럼 우선 CUDA환경에서 LM(language model), 비전, 데이터프레임, 시뮬레이션 등 다양한 테스트를 한번에 진행할 수 있도록 제작해봐. 저장소(로컬에서는 SSD역할)는 구글 드라이브를 활용하도록 제작해. 다만, CPU환경과 T4 GPU 환경 중 테스트에 더 적합해보이는 환경을 선택해."

## 실행 방법

- 노트북 `examples/colab_t4_suite.ipynb`(셀 하나)를 **Colab T4 런타임**에서 실행한다.
- 준비: Drive `memopro_colab/install/`에 소스 묶음 `memopro-src.tar.gz`(main의 `git archive`)를 올린다. 모델 약 26GB는 Drive `hf_cache/`에 남는다.
- 모든 경우는 새 프로세스에서 돈다. 결과는 Drive `memopro_colab/results/suite/<실행 id>/`에 남고, 끊겨도 이어서 한다(0091 구조).
- 노트북의 측정 코드: `examples/colab_t4/cell_suite.py`, `worker_suite.py`, `worker_data.py`. 노트북에 적힌 커밋(`EXPECTED_COMMIT`)과 `env.json`의 `build_has.0195_cuda_stream`을 함께 기록한다.

## 경우

| 갈래 | 모델·작업 | 경우 |
|---|---|---|
| 언어 모델 | Qwen2.5-3B-Instruct (호스트 예산 2·3GiB), Qwen2.5-7B-Instruct (4·6GiB) | 생성(프롬프트 2개 × 32토큰, 탐욕): 그냥 GPU / memopro 두 예산. 16비트 LoRA(r 8, q·k·v·o, 5단계 × 512토큰, WikiText-2): 그냥 GPU(PEFT + 체크포인팅) / memopro `finetune` 두 예산 |
| 비전 | ResNet-152 (60·120MiB), DINOv2-giant (1.1·2.2GiB, 모델의 약 1/4·1/2) | 추론(무작위 영상 8장, 고정 시드): 그냥 GPU / memopro 두 예산. DINOv2 LoRA(query·value, 새 선형 머리, 5단계 × 8장): 그냥 / memopro 두 예산 |
| 데이터 (CPU) | E027 영상 묶음, E040 pandas·scikit-learn·열 확산 | 그냥 / `memopro-preload`(페이저 예산 = 그냥 실행 최대 RSS의 1/2, 프로세스 예산 = 그 − 64MiB). 먼저 preload 시험(C)으로 userfaultfd가 되는지 확인 |

- LoRA 경우는 `CUBLAS_WORKSPACE_CONFIG=:4096:8`과 `torch.use_deterministic_algorithms(True, warn_only=True)`로 돈다.
- Colab에는 cgroup 한도가 없다. 그래서 데이터 작업은 "한도에서 그냥 실행이 죽는다"를 보이지 못하고, 프로세스 최대 RSS로 상한을 본다.

## 기준 (`summary.md`가 자동 판정)

| # | 기준 |
|---|---|
| **G-모델** | 그냥 실행이 되면: memopro 두 예산의 생성 글이 그냥 실행과 같다. 그냥 실행이 T4에 안 들어가면(7B 예상): memopro가 두 예산에서 완주하고 글이 서로 같다 |
| **L-모델** | memopro LoRA가 두 예산에서 5단계를 끝내고 손실이 비트 동일 |
| **V-모델** | memopro 추론 출력(해시)이 두 예산 모두 그냥 실행과 같다 |
| **VL-DINOv2** | memopro 비전 LoRA가 두 예산에서 끝나고 손실이 비트 동일 |
| **D-작업** | preload 시험이 통과한 경우에만 판정: 결과가 그냥 실행과 같고, 프로세스 최대 RSS ≤ 한도 + 64MiB, 페이저 overruns = 0 |

- **보고만**: 속도(초/토큰, 단계 시간, 추론 시간), 그냥 LoRA 손실과 memopro 손실의 차이(LoRA 초기화 난수가 달라 같지 않을 수 있다), GPU·호스트 최대 메모리.
- **채점표(0179)에 주는 뜻**
  - G·L(7B) 통과 → A14(CUDA에서 G4 경로) ✔, U16(CUDA 런타임 경로) ✔.
  - V·VL 통과 → U5(비전 학습)·U6(비전 추론) ✔.
  - D: Colab에서는 cgroup 판정(T2)이 없으므로 채점 조건(B9·B10·U8·U11·U12)은 E040b(Linux CI)로 판정한다. 여기서는 보고한다.
- 실패하면 멈추고 보고한다. 기준은 바꾸지 않는다.

## 이 Mac에서의 점검 (판정 아님)

- 셀 로직을 작은 설정으로 돌렸다(`MP_LOCAL_SMOKE`, 장치 MPS, Qwen2.5-1.5B, ResNet-152, 영상 묶음).
- 생성 글이 그냥 로딩과 같고, LoRA가 두 예산에서 비트 동일, ResNet 출력이 같았다.
- 비전 LoRA는 DINOv2-small로 따로 확인했다. 두 예산에서 비트 동일, `query`·`value`에 어댑터가 붙었다.
- preload는 Linux 전용이라 건너뛰었다.

## 논문 매핑

- **논문 P1·P3 Evaluation**: 분리 메모리 GPU(T4)에서 7B 16비트 LoRA·생성, 비전 모델 흘려 쓰기, 수정 없는 데이터 프로그램.
