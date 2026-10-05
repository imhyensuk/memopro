# 0195. Colab 통합 시험의 환경 선택(T4 GPU)과 제작: CUDA로 흘려 쓰기, 비전 모델 흘려 쓰기, 생성 설정, Drive 저장소

- **날짜**: 2026-10-05
- **유형**: decision + implementation
- **상태**: 확정. E042(0197)로 검증
- **관련 기록**: 0179 (채점표: A14 CUDA 경로, U5·U6 비전, U8·U11·U12 데이터, U16 CUDA 런타임), 0189·0196 (memopro-preload), 0091·0092 (Colab 노트북 구조)

## 사용자 지시

> "그럼 우선 CUDA환경에서 LM(language model), 비전, 데이터프레임, 시뮬레이션 등 다양한 테스트를 한번에 진행할 수 있도록 제작해봐. 저장소(로컬에서는 SSD역할)는 구글 드라이브를 활용하도록 제작해. 다만, CPU환경과 T4 GPU 환경 중 테스트에 더 적합해보이는 환경을 선택해."

## 환경 선택: T4 GPU

| | Colab CPU 런타임 | Colab T4 런타임 |
|---|---|---|
| CPU·호스트 메모리 | 2 vCPU, 약 12.7GB | 같음 |
| GPU | 없음 | T4 15GB |
| 언어 모델·비전 | CPU 계산만(bf16 backward가 느림, 0115) | GPU 계산. 7B bf16(15.2GB)은 T4에 그냥은 안 들어가 흘려 쓰기의 시험대가 됨 |
| 데이터 작업 | CPU | CPU(같음) |
| 단점 | — | GPU 사용 시간 제한(무료 등급) |

- **T4를 고른다.** CPU·메모리는 같고 GPU가 더해진다. 그래서 CPU 런타임에서만 할 수 있는 시험이 없다.
- 데이터 작업(pandas·scikit-learn·시뮬레이션)은 T4 런타임에서도 CPU로 돈다.

## 저장소: Google Drive

- 기존 Colab 구조(0091)를 그대로 쓴다. Drive `memopro_colab/`의 하위 폴더는 다음과 같다.
  - `install/`: 소스 묶음
  - `hf_cache/`: 모델(세션이 바뀌어도 남는다)
  - `results/suite/<실행 id>/`: 경우별 JSON, 로그, 시간축, 요약
- 모델은 Drive 캐시에서 로컬 디스크로 한 번 복사해 쓴다(`STAGE_TO_LOCAL`). Mac에서 SSD가 하는 역할이다.
- Drive 공간이 모자라면 `MODEL_CACHE = "local"`(그 세션에만 받기)로 바꾼다.
- **설치는 소스 묶음 `memopro-src.tar.gz`로 한다**(저장소 전체의 `git archive`). Colab에서 memopro(maturin)와 `memopro-preload`(cargo)를 함께 빌드한다.
  - 저장소가 비공개라 Colab이 직접 받을 수 없다.
  - 예전 sdist에는 preload 크레이트가 없다.
  - `GITHUB_TOKEN` 보안 비밀이 있으면 저장소를 받아 같은 방식으로 빌드한다.

## 제작

### 1. CUDA로 흘려 쓰기 (`stream_model(device="cuda")`)

- Mac(MPS)은 통합 메모리라 런타임 메모리를 GPU가 그대로 쓴다. T4는 GPU 메모리가 따로 있다.
- **복사 경로**(`StreamedWeights.copy`)
  - 모듈의 forward 직전에 런타임 버퍼를 고정하고 GPU로 복사한 뒤 고정을 바로 푼다.
  - backward에서 필요한 가중치는 `saved_weights`가 다시 고정해 다시 복사한다.
  - GPU에는 한 모듈의 가중치와 활성값만 있다. `budget`은 호스트 메모리다.
- **CUDA에서는 활성값 몫을 덜어 두지 않는다**(`finetune`·`generate`). 활성값과 KV 캐시는 GPU 메모리에 있어 호스트 예산과 상관없다.
- `memopro.finetune`/`generate`의 `device="auto"`는 CUDA → MPS → CPU 순으로 고른다.
- 이 Mac에는 CUDA가 없다. 그래서 복사 경로를 CPU에서 강제로 켜는 시험을 넣었다(`test_copy_path_trains_like_weights_in_place`). 제자리 경로와 LoRA 손실이 비트 동일하고 고정이 남지 않는다.

### 2. 비전 모델 흘려 쓰기

- transformers 5는 일부 모델(ViT)의 체크포인트 이름을 불러올 때 바꾼다(`encoder.layer.N.attention.attention.query` → `layers.N.attention.q_proj`).
- `stream_model`이 transformers의 변환표(`get_model_conversion_mapping`)에서 **이름만 바꾸는 규칙**을 따르게 했다(`_renamed`). 텐서를 합치거나 나누는 규칙이 필요한 키는 이름을 그대로 두어, 빠진 가중치로 보고된다.
- 확인(작은 무작위 모델, CPU): ViT 분류기(예산 1/8), ResNet(CNN), DINOv2가 모두 일반 실행과 출력 비트 동일. 시험 `test_streamed_vision_model_matches_plain`.

### 3. 생성 설정

- `stream_model`이 `generation_config.json`도 읽는다(Qwen2.5: `repetition_penalty` 1.05). 그래야 일반 `from_pretrained` 모델과 생성 결과를 비교할 수 있다.
- 이전 실험(E033~E041)은 흘려 쓰는 모델끼리 비교했으므로 영향이 없다.

### 4. Colab 노트북 `examples/colab_t4_suite.ipynb`

- 기존 조립기(`build.py`)에 노트북별 추가 파일(`EXTRA_WORKERS`)을 넣었다. 데이터 작업 파일, preload 시험 C 코드, 새 작업자 `worker_suite.py`(언어 모델·비전), `worker_data.py`(데이터 작업을 그 프로세스에서 그대로 실행)가 노트북에 들어간다.
- 경우와 판정은 0197.

## 논문 매핑

- **System**: 통합 메모리(무복사)와 분리 메모리(복사)에서 같은 런타임으로 흘려 쓰기. 비전 모델까지 같은 경로.
