# 0052. v0.2·v0.3을 지금 제작한다: 범위, 사전 등록한 완료 조건, 설계 결정

- **날짜**: 2026-09-26
- **유형**: decision
- **상태**: 확정
- **관련 기록**: 0036 (Gβ 전 제작 선례), 0013 V15 (A2 완료 조건은 착수 전 확정), 0030 C4·C5, 0051 (재조사), development-plan A2·N3

## 배경

> "아직 만들지 않은 것 제작을 전부 진행해."

계획상 남은 것은 다음과 같다.
- **A2(v0.2)**: `load`·`optimize`·`train_session`·`check`, 후보 구성 선택, census 정밀 모드, 기존 기법 연동
- **N3(v0.3)**: Rust `pressure`, γ, `memopro run`
- **자잘한 것**: `idle_seconds`, Windows 방출 쓰기, 풀별·비율 예산

계획의 순서는 v0.1 배포 → 사용자 피드백 → 관문 Gγ → N3였다.

## 결정

### E1. 순서: 사용자 지시로 Gγ 전에 제작한다 (0036과 같은 방식)

- v0.2와 v0.3을 지금 만든다. Gγ(v0.2 피드백)는 사용자가 없어 지금 판정할 수 없으므로 **v0.3 배포 전 확인**으로 옮긴다.
- 버전 번호는 그대로 둔다(0.1.0a1, 미배포). 첫 공개 버전을 몇으로 할지는 사용자 결정으로 남긴다.
- 작업 단위: v0.2를 PR 하나로, v0.3을 PR 하나로 올린다. CUDA 전용 확인은 Colab 4차에서 한다.

### E2. A2 완료 조건 (사전 등록, 0013 V15)

이 Mac은 사용 가능 RAM 1.44GiB, 디스크 여유 19GB, OS 압박 "경고" 상태다(0051). 원래 조건 "M1에서 MPS 한도를 넘는 3B 모델"은 지금 확인할 수 없다. 그래서 조건을 **로컬(작은 모델 + 인위적 예산·장치 한도)**과 **Colab 4차(실제 규모)**로 나눈다. 원래 조건은 열린 과제로 남긴다.

| # | 조건 | 판정 |
|---|---|---|
| A2-1a | 로컬: GPT-2 small(fp32 498MB)을 `load(..., budget=...)`로 예산 아래에서 불러온다. 예산이 bf16을 허용하면 bf16, 더 작으면 int8을 고르고 생성이 동작한다 | 고른 구성의 가중치 바이트 ≤ 예산, 생성 성공, 선택 이유가 `report()`에 있음 |
| A2-1b | Colab: fp16 크기가 T4 장치 예산을 넘는 7B급 모델을 `load`가 int8·int4(bitsandbytes)로 불러와 생성한다 | 장치 최대 할당 ≤ 장치 예산, 생성 성공 |
| A2-1c | fail-open: 첫 구성의 적용을 강제로 실패시키면 다음 구성으로 다시 불러온다 | 실패가 기록되고 다음 구성이 쓰임 |
| A2-2 | 장치 한도(`set_per_process_memory_fraction`) 아래에서 보통 학습 스텝이 OOM인 설정을 `train_session`으로 끝까지 학습한다 | N스텝 완주. 한도 없는 같은 유효 배치의 기준 실행과 최종 파라미터의 최대 상대 차이 ≤ 1e-5(fp32, 합산 순서 차이만 허용) |
| A2-3 | `check`의 최대 메모리 예측 | CUDA `max_memory_allocated` 대비 ±15% 이내. GPT-2 small·medium, 추론(배치 1, 길이 1024)·학습(AdamW, 배치 2·8, 길이 512), Colab 4차 |
| A2-4 | 기존 도구와 같은 과제 비교(0030 C4) | `check` 대 `accelerate estimate-memory`, `train_session` 대 accelerate `find_executable_batch_size`(OOM 횟수, 첫 성공까지 시간, 결과 일치). 우위가 없으면 그대로 적는다 |

### E3. 접근 계층 API

- **품질 등급**(U4·U5): `quality`로 자동 적용을 허용할 최대 손실을 정한다.

  | 값 | 자동 적용 범위 |
  |---|---|
  | `"lossless"` | 정확한 기법만 |
  | `"high"` | + 반정밀도(fp16/bf16) |
  | `"balanced"`(기본) | + int8 가중치 |
  | `"low"` | + int4 가중치 |

- **선호**: `prefer="speed"`(기본) | `"quality"` | `"memory"`.
  - 기본 `speed`는 설계 §3.3 표의 순서를 따른다(반정밀도 → int8 → int4 → CPU 오프로드 → 디스크 오프로드).
  - `quality`는 손실 없는 오프로드를 양자화보다 먼저 고른다.
  - `memory`는 가장 작은 구성부터 고른다.
- **`load(model_id, *, task=None, budget, quality, prefer, allow, deny, tokenizer=False)`**: 모델을 돌려준다. `tokenizer=True`이면 `(model, tokenizer)`를 돌려준다.
  - 가중치를 받기 전에 메타데이터(config, safetensors 헤더)로 크기를 계산한다.
  - 로드 시점형 구성은 실패하면 원본에서 다음 구성으로 다시 불러온다(§3.4).
- **`optimize(model, goal)`**: 이미 불러온 모델에 실행 시점형 기법만 적용한다.
  - 추론: 품질 범위 안의 dtype 변환·torchao 양자화, 분리형 GPU의 오프로드.
  - 학습: 활성값 체크포인팅.
- **`train_session`**: 두 가지 길을 둔다.
  - `s.step(batch, loss_fn)`: forward·backward·optimizer step을 memopro가 수행한다. OOM이 나면 그 배치의 그래디언트를 버리고 마이크로배치를 더 잘게 나눠 **처음부터 다시** 한다. 결과는 합산 순서만 다르다.
  - `for mb in s.batches(loader): s.backward(loss)`: 설계 §4.1의 형태다. 사용자가 forward를 하므로 OOM 재시도는 없다. 대신 처음부터 예산에 맞춘 마이크로배치를 쓴다.
  - **옵티마이저 교체(8비트, CPU 오프로드)는 제안만 한다.** 사용자가 들고 있는 옵티마이저 참조가 무효가 되기 때문이다. 설계 표의 "자동 적용"에서 벗어나므로 여기 기록한다. LoRA 전환은 설계대로 제안만 한다.
  - BatchNorm 등 배치에 의존하는 층을 감지하면 마이크로배치가 정확하지 않다고 경고한다.
- **`check(target)`**: 모델 ID, 로컬 폴더, `nn.Module`을 받는다.
  - 학습 메모리는 torch `MemTracker` + `FakeTensorMode`로 **할당 없이** 추정하고, 쓸 수 없으면 공식으로 추정한다.
  - **스크립트 경로는 받지 않는다.** 이유를 알리고 `memopro run --dry-run`을 안내한다. 설계 §4.2의 "스크립트"에서 벗어나므로 기록한다.
- **예산 형식**: `"auto"` | 크기(`"6GB"`) | 비율(`0.5`, 계산된 예산에 곱함) | 풀별(`{"device": "4GB", "host": "6GB"}`, 환경변수는 `device=4GB,host=6GB`).

### E4. 연동 백엔드 (재구현 금지, 0010)

| 기법 | 백엔드 | 가용성 판정 |
|---|---|---|
| 지연 로딩, dtype | transformers `from_pretrained` | 항상 |
| int8·int4 가중치 | bitsandbytes 우선, 없으면 torchao | **실제 작은 층으로 시험해 판정**(torchao int4는 로컬에서 `mslk`가 없어 불가, 0051) |
| CPU·디스크 오프로드 | accelerate `device_map`·`max_memory`·`offload_folder` | 디스크 오프로드는 SSD에 쓰므로 `disk_writes` 정책을 따른다 |
| 체크포인팅 | HF `gradient_checkpointing_enable`, 일반 모듈은 `torch.utils.checkpoint` | 항상 |
| 혼합 정밀도 | `torch.autocast` | 장치별 |
| 활성값 오프로드 | `torch.autograd.graph.save_on_cpu` | 분리형 GPU만 |
| 제안만 | 8비트 옵티마이저(bitsandbytes·torchao), LoRA(peft) | — |

- 이번에 **넣지 않는 것**: KV 캐시 양자화(quanto·HQQ 필요, 미설치), Diffusers·TRL 통합. 확인할 수 없는 연동은 만들지 않고 남은 과제로 기록한다.

### E5. census 정밀 모드 (0030 C3 정의)

- API: `census.record(model, optimizer, mode="deep", probe=fn)`. `fn()`은 고정 배치의 손실(스칼라)을 돌려준다. 정밀 모드에서는 필수다.
- 블록이 끝나면 범주(가중치·그래디언트·옵티마이저 상태·저장 활성값) × 비트 폭(2·4·8·16)마다 교란한다. 범주 전체를 경량판과 같은 블록 absmax 양자화로 교란하고, 교란하지 않은 기준과 비교한다.
  - 가중치: 손실 상대 변화, 그래디언트 코사인
  - 그래디언트·옵티마이저 상태: optimizer step 한 번의 파라미터 갱신량 코사인과 상대 오차. 상태는 저장했다가 되돌린다
  - 저장 활성값: `saved_tensors_hooks`로 교란한 채 역전파한 그래디언트 코사인
- 필요 비트 = 기준(손실 상대 변화 ≤ 1e-3, 코사인 ≥ 0.999, 조정 가능)을 만족하는 가장 작은 비트. **낭비 = 저장 비트 − 필요 비트**. 가능한 절감 바이트와 함께 보고한다.
- 끝나면 모델·옵티마이저 상태를 비트 단위로 되돌린다(테스트로 확인).

### E6. γ 설계

- **Rust `pressure::current()`**:
  - macOS: `kern.memorystatus_vm_pressure_level`(1 정상, 2 경고, 4 위험)
  - Linux: 자기 cgroup v2의 `memory.pressure`, 없으면 `/proc/pressure/memory`의 some·full avg10. 초기 기준(보정 예정): 경고 = some ≥ 10 또는 full ≥ 1, 위험 = some ≥ 40 또는 full ≥ 10
  - 그 밖의 OS: `NotSupported`(fail-open)
- **감시 스레드는 읽기만 한다.** 백그라운드 스레드가 사용 중인 텐서를 동면시키면 메인 스레드의 계산과 충돌하기 때문이다. 조치는 **안전 지점**에서만 한다.

  | 안전 지점 | 압박 시 조치 (쓰기 없는 것부터) |
  |---|---|
  | 노트북 셀 경계 | 캐시 비우기 → 유휴 객체 동면(`source`·`host`·`compress`) → `spill`은 `disk_writes="allow"`일 때만(배경에서는 묻지 않음) |
  | `train_session` 스텝 경계 | 마이크로배치 절반 → 체크포인팅 → 활성값 오프로드(CUDA). 정상이 `up_after`초 이어지면 한 단계씩 되돌림. 모두 정확한 기법 |
  | `load` | 압박 중이면 예산을 줄여 더 작은 구성 선택 |
  | 사용자 루프 | `memopro.elastic.checkpoint()` 호출 지점 |

- API: `memopro.elastic.enable(interval=1.0, up_after=30)`, `disable()`, `status()`, `current()`, `checkpoint()`.
- 평가는 별도 사전 등록 실험으로 한다. 기준선은 Tri-Accel식 자기 VRAM 되먹임과 accelerate OOM 재시도다(0051).

### E7. `memopro run`

- `memopro run [--budget] [--quality] [--disk-writes] [--modes] [--no-elastic] [--census] [--dry-run] script.py args…`: 스크립트를 같은 프로세스에서 실행한다(`runpy`).
- **이 모드에서만 전역 패치를 한다(P4 예외, `report()`에 표시)**:
  1. transformers `from_pretrained` 로딩 정책. 호출자가 `device_map`·`quantization_config`·`dtype`·`max_memory`를 하나도 주지 않았을 때만 `load`와 같은 선택을 적용한다. **명시한 선택은 절대 바꾸지 않는다**
  2. γ 감시
  3. (선택) census 빠른 모드
- 끝나면 요약을 표준 오류로 출력한다. `--dry-run`은 무엇을 패치할지 보여주고 실행하지 않는다.

### E8. 자잘한 것

- `idle_seconds`: 셀 경계에서 관찰한 마지막 사용 시각 기준. 셀 수와 시간 중 하나라도 넘으면 제안한다.
- Windows 방출 쓰기: Rust `seek_read`/`seek_write`, 파일 권한은 사용자 폴더 기본 ACL, 저우선순위 I/O는 없음(문서화). Python의 프로세스 생존 확인은 Windows에서 `os.kill(pid, 0)`을 쓰지 않는다(신호 0이 CTRL_C_EVENT가 된다).
  - **CI에 Windows 작업을 추가한다.** macOS처럼 main 푸시와 수동 실행 때만 돈다(0043 정책 유지).

### E9. 신규성

- 접근 계층(E3·E4·E7)은 신규성을 주장하지 않는다.
- γ는 "not found in our survey"(0051)로만 적는다.
- census 정밀 모드는 0030 C3의 범위를 지킨다.

## 논문 매핑

- **논문 B System**: 품질 등급과 선호에 따른 후보 구성 선택, 안전 지점에서만 행동하는 γ, 전역 패치를 명시 모드로 한정한 `run`.
- **논문 B Evaluation**: E2의 사전 등록 조건과 기존 도구 비교(A2-4).
- **연구 방법론**: 하드웨어 제약(가용 1.44GiB)에 맞춰 완료 조건을 로컬·Colab으로 나누고 원래 조건을 열린 과제로 남긴 기록.
