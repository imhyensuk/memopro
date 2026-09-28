# 0091. E021(Colab 7차) 준비와 사전 등록: T4에서 무거운 모델로 학습·추론·여러 모델 재검증

- **날짜**: 2026-09-29
- **유형**: experiment (사전 등록) + tool (노트북 제작)
- **상태**: 확정. 실행은 사용자가 Colab에서 한다. 결과는 다음 기록
- **관련 기록**: 0090 (완성도 평가: 마지막 CUDA 검증이 0056), 0054~0056 (Colab 4~6차), 0055 (측정은 경우마다 새 프로세스), 0084·0085 (int4 품질, 긴 입력), 0088·0089 (γ)

## 요청

> "연구 기록으로 남기고 Colab 재검증 준비해. Colab T4 GPU 환경에서 무거운 모델로 시험해보자. 결과 로그나 결과 파일들이 상세하게 잘 도출될 수 있도록 검토해서 제작하고, 하나의 통합 셀 구조로 제작해. 또한 구글 드라이브 용량이 400GB 정도로 넉넉하니까 구글 드라이브 저장공간을 적극적으로 활용할 수 있도록해. 동시에 여러 모델을 테스트, 실험해볼 수 있는 구조여도 좋을 것 같아. 이 라이브러리의 성능을 아주 객관적으로 볼 수 있도록 하고, 단순히 로컬 AI 모델 구동뿐만 아니라 트랜스포머 등의 딥러닝 학습 과정에서 어떤 성능을 발휘하는지도 테스트할 수 있도록 제작해. 그래서 총 학습 및 개발 과정에 대한 테스트 통합 셀 1개, 로컬 AI 모델 활용 테스트 통합 셀 1개를 기본으로 제작하고, 너가 판단하기에 추가적으로 Colab 환경에서 테스트가 필요한 것이 있다면 통합 셀 형태로 만들어."

## 1. 노트북 (`examples/colab_t4_heavy.ipynb`, 원본 `examples/colab_t4/`)

- **구조**: 셀마다 "설정 + 공통 코드 + 워커 원본 + 본문"을 담은 **독립 셀 3개**다.
  - 어느 셀이든 하나만 실행해도 된다.
  - `build.py`가 `common.py`, `worker_*.py`, `app_naive_load.py`, `cell_*.py`를 조립한다. 저장소 시험 `tests/test_colab_notebook.py`가 조립과 컴파일, 커밋된 노트북이 원본과 같은지를 확인한다.
- **셀 1 학습·개발**
  - 모델: GPT-2(124M), Qwen2.5-0.5B, Qwen2.5-1.5B.
  - 전체 미세조정(fp32 AdamW, WikiText-2 train 실제 토큰, dropout 끔, 같은 시드·데이터·단계).
  - 시나리오
    - `plain`(기준).
    - `plain_amp`(fp16 autocast + GradScaler).
    - `hf_ckpt`(HF gradient checkpointing).
    - `accel_find_batch`(accelerate 배치 탐색, 배치가 줄 수 있음).
    - `memopro`(`train_session`), `memopro_lossless`.
    - `check`(`memopro.check` 예측 대 실측 정상 상태 피크).
  - 작은 GPU 흉내: `set_per_process_memory_fraction`(GPT-2 30%, 0.5B 50%). memopro에는 같은 크기를 예산으로 준다(그 크기의 장치를 재는 상황을 흉내).
  - 7B QLoRA: 표준 방식(bnb nf4 + peft + gradient checkpointing) 대 `memopro.load(quality="low")` + peft + `train_session`.
- **셀 2 로컬 AI 모델**
  - 모델: Qwen2.5 1.5B·3B·7B·14B Instruct.
  - 방법: fp16 `device_map="auto"`(CPU·디스크 넘김 포함), bnb 8비트·4비트, `memopro.load` 기본(balanced)과 `quality="low"`.
  - 14B fp16은 기본으로 건너뛴다(28GB > GPU 15GB + 호스트 12.7GB).
  - 측정
    - 불러오기 시간, 가중치 위치(장치별 바이트, `hf_device_map`).
    - GPU·호스트 최대 사용.
    - 512·2048토큰 프롬프트의 첫 토큰 시간.
    - 128토큰 디코딩 속도.
    - WikiText-2 test perplexity(8 × 1024토큰).
    - 채팅 질문 3개의 탐욕 답변(토큰 id). 기준 대비 일치와 공통 접두 비율을 본다.
  - 기준은 GPU에 다 올라간 가장 정확한 방법이다(fp16, 없으면 8비트).
- **셀 3 여러 모델·프로세스 수준**
  - `doctor` 예산 대 CUDA free·MemAvailable.
  - 모델 3개(Qwen2.5-3B, Phi-3.5-mini, Qwen2.5-1.5B, fp16 합 약 17GB > T4)를 2바퀴 번갈아 쓰기. 방법은 셋이다.
    - 모두 올려 두기.
    - 지우고 다시 불러오기.
    - memopro 동면. `disk_writes="allow"`, spill은 로컬 디스크.
    - 같은 질문의 답이 바퀴마다 같은지 확인한다.
  - 수정하지 않은 스크립트(`from_pretrained(path)`만 부름)를 `python` 대 `memopro run`으로 돌린다(1.5B, 7B).
    - `python`이 호스트 RAM의 90%를 넘길 것으로 예상되면 돌리지 않고 이유를 남긴다. OOM killer가 노트북 커널을 죽일 수 있기 때문이다.
  - 최근 변경의 CUDA·Linux 회귀 검사 7종
    - int4 그룹 64(bnb).
    - 품질 비용 문구.
    - `residency="file"` 거절.
    - 플랫폼 기본값(MPS 안내 없음, malloc 안내 없음, `run`의 γ 켜짐).
    - PSI 읽기.
    - 동면 비트 동일성.
    - `check` 추론 예측.
- **Drive 활용** (`MyDrive/memopro_colab/`)
  - `install/`: sdist나 wheel.
  - `hf_cache/`: 모델·데이터 캐시. 세션이 바뀌어도 남는다. 불러올 때는 로컬 디스크로 복사해 Drive 읽기 속도를 재지 않는다.
  - `results/<셀>/<실행 id>/`
    - `summary.md`·`summary.csv`: 읽는 법 포함.
    - `cases/*.json`: 경우마다 손실 곡선, 단계 시간, 메모리, memopro 보고, 오류와 추적.
    - `logs/*.log`: 프로세스 출력 전체.
    - `timelines/*.csv`: 0.5초 간격 nvidia-smi 메모리·사용률, 호스트·스왑.
    - `env.json`: 판, GPU, 빌드 포함 여부 `build_has`.
    - `progress.jsonl`, `.zip`.
- **이어하기**
  - 경우마다 결과를 바로 Drive에 저장한다. 다시 실행하면 끝난 경우는 건너뛴다. 멈춤과 시간 초과는 다시 돈다.
  - 설정이 바뀌면 새 실행으로 시작한다.
- **여러 모델**: 모델 목록 × 방법 목록을 차례로 돈다. 한 GPU에서 동시에 재면 서로의 측정을 오염시키므로(0085의 3B 사례) 동시에 돌리지 않는다.
- **설치**
  - 우선순위: Drive `install/`의 Linux wheel → sdist(`memopro-0.1.0a1.tar.gz`, Colab에서 Rust를 설치해 빌드, 3~5분) → 사용자가 직접 등록한 Colab 보안 비밀 `GITHUB_TOKEN`으로 저장소에서 설치.
  - 로컬 zig 교차 빌드(0049)는 zig가 없어 쓰지 않았다. 설치하려면 다운로드 승인이 필요하다.
  - sdist는 main `701a260`(0089)의 라이브러리 코드로 만들었다(노트북의 `EXPECTED_COMMIT`).

## 2. 로컬 확인 (측정이 아님)

- 조립된 셀을 이 Mac(MPS)에서 작은 설정으로 끝까지 돌렸다(`MP_LOCAL_SMOKE=1`).
  - 학습: GPT-2, 배치 2, 64토큰, 4단계.
  - 추론: Qwen2.5-1.5B, 3가지 방법.
  - 여러 모델: GPT-2 + Qwen2.5-1.5B.
- **고친 것**
  1. **MPS에서 `tensor.to("cpu", torch.float64)`가 조용히 틀린 값(0과 nan)을 냈다**(torch 2.14). 학습 결과 비교용 노름을 `.cpu().double()` 두 단계로 바꿨다. memopro 라이브러리 코드에는 이 패턴이 없다(`python/memopro`를 검색해 확인).
  2. 동면 핸들의 `disk_write_bytes`는 속성, `bytes_by_mode()`는 메서드였다.
  3. macOS의 `ru_maxrss` 단위(바이트, Linux는 KB).
  4. `/content`, `/proc/meminfo`가 없는 환경의 대체 경로. spill·offload 폴더를 작업 폴더 아래로 옮겼다.
  5. 채팅 템플릿이 없는 모델의 대체 경로.
- **로컬 결과** (설계 확인용, 판정에 쓰지 않음)
  - GPT-2에서 `memopro`·`memopro_lossless`는 `plain`과 손실 차이 1e-6, 파라미터 지문 차이 0이었다. AMP는 손실 차이 0.12였다.
  - 동면 번갈아 쓰기의 두 번째 바퀴 전환은 중앙값 3.5초, 다시 불러오기는 16.2초였다. 답은 모두 반복됐다.
  - 회귀 검사는 macOS에서 4/7이었다. 실패한 셋은 CUDA·Linux 전용 기대값이라 예상대로다.

## 3. 판정 기준 (실행 전 고정)

| # | 질문 | 기준 |
|---|---|---|
| R1 | 최근 변경이 CUDA 경로를 깨지 않았는가 | 셀 3 회귀 검사 7종이 모두 통과(PSI는 읽히지 않으면 보고만). **배포 전 필수** |
| T1 | `train_session`은 같은 모델을 학습하는가 | `plain`이 성공한 경우, `memopro`의 손실 차이 ≤ 1e-3(상대) 그리고 파라미터 지문 상대 차이 ≤ 1e-5 |
| T2 | 작은 GPU에서 쓸모가 있는가 | 상한 걸린 경우 가운데 `plain`이 OOM인 곳에서 `memopro`가 끝까지 학습한다(보고: 속도 비용, `hf_ckpt`·accelerate와 비교) |
| T3 | `check` 정확도 | 실측이 있는 경우 |오차| ≤ 15%(A2 기준, 0052). 실측이 OOM이면 `check`가 "맞지 않음"이라고 했는지 |
| T4 | 들어가지 않는 학습 | 1.5B 전체 미세조정처럼 아무 방법도 안 되는 경우, memopro가 OOM 대신 `BudgetExceeded`와 제안(8비트 옵티마이저, LoRA)을 낸다 |
| I1 | 추론 불러오기 | `memopro`·`memopro_low`는 OOM으로 끝나지 않는다: 불러오거나(`ok`), 제안과 함께 거절한다(`does_not_fit`) |
| I2 | 품질·속도 | 보고만 한다(PPL 차이, 답변 일치, 첫 토큰 시간, 디코딩 속도, 메모리). 같은 기법의 비교(memopro int4 대 bnb4 등)는 같은 표에서 한다 |
| M1 | 동면의 정확성 | `rotate: hibernate`에서 모든 답이 반복된다(비트 동일 복원의 결과) |
| M2 | 동면의 가치 | 보고: 전환 시간과 전체 시간을 `reload`·`keep_all`과 비교 |
| M3 | `memopro run` | 7B를 코드 수정 없이 불러와 답한다. 1.5B처럼 원래대로 들어가면 개입하지 않는다(보고에 "fits as stored") |

- R1이 실패하면 배포 전에 고친다(결함 기록).
- T1이 실패하면 `train_session`의 정확성 주장을 멈추고 원인부터 찾는다.
- 나머지는 결과 기록에서 그대로 보고한다.

## 한계 (미리 적어 둠)

- T4 하나, Colab 무료·유료 환경의 호스트 RAM(약 12.7GB)과 디스크에 묶인다.
- 작은 GPU는 `set_per_process_memory_fraction`으로 흉내 낸다. 실제 작은 GPU와 할당기 동작이 다를 수 있다.
- 학습은 짧다(10~30단계). 수렴 품질이 아니라 같은 계산인지와 메모리·속도를 본다.
- 품질은 WikiText-2 PPL과 3개 답변뿐이다. 과제 정확도는 재지 않는다.
- Qwen2.5 계열 중심이다(Phi-3.5-mini 하나 추가). gated 모델은 쓰지 않는다.

## 논문 매핑

- **논문 B Evaluation**: CUDA에서의 객관 비교 표(학습: 일반·AMP·checkpointing·accelerate 대 memopro. 추론: HF 기본·bnb 대 memopro. 여러 모델: 다시 불러오기 대 동면). 결과 기록에서 채운다.
- **논문 B Artifact**: 재현 가능한 단일 셀 노트북과 Drive 결과 구조.
