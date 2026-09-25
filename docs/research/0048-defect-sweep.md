# 0048. 결함 소탕: 탐색적 시험 3차례, 결함 16건 수정, 가드 설계

- **날짜**: 2026-09-25
- **유형**: survey+milestone (탐색적 시험, 결함 수정)
- **상태**: 확정
- **관련 기록**: 0041 (D1~D3), 0036 B1 (정체성 유지 해제), 0047 (검증 종합)

## 배경

> "결함을 최대한 줄여봐."

0047에서 확인한 사실이 있다. **조용한 오류는 자동 테스트보다 탐색적 시험에서 드러났다.** 그래서 동면 중인 객체에 닿을 수 있는 경로를 범주별로 체계적으로 시도했다. 스크립트 3개로 총 40종을 시험했다(작업 공간 `probe_defects.py`, `probe_round2.py`, `probe_round3.py`. 결과는 회귀 테스트로 옮겼다).

## 1차: 동면 중인 객체에 닿는 경로 (15종 → 결함 10건)

| # | 경로 | 증상 | 심각도 |
|---|---|---|---|
| S1a | `model.state_dict()` | **빈 텐서를 조용히 반환** | 🔴 조용함 |
| S1b | `torch.save(model)` | 내부 hook 함수 때문에 pickle 실패 | 🟠 |
| S1c | `copy.deepcopy(model)` | **빈 복사본** | 🔴 조용함 |
| S1d | `pickle.dumps(tensor)` | **원소 0개 텐서가 저장됨** | 🔴 조용함 |
| S1e | `load_state_dict()` | 크기 불일치 오류. 불러온 값을 쓸 수 없음 | 🟠 |
| S2 | `.double()`/`.to()` | 깨울 때 원래 dtype으로 되돌아가 dtype이 섞임 | 🟠 |
| S3 | 동면 도중 Ctrl-C | **핸들 없이 빈 텐서가 남음(데이터 손실)** | 🔴 |
| S4a | 깨운 모델을 `del` | 레지스트리가 모델을 붙잡음(**누수**) | 🟠 |
| S4b | 동면 중인 모델을 `del` | 모델과 복원용 데이터가 남음(**누수**) | 🟠 |
| S5 | 여러 스레드가 동면 중인 모델 호출 | 동시 복원 경쟁 → 형상 오류 | 🟠 |
| S7 | 동면 중 `optimizer.step()` | **스텝이 조용히 건너뛰어짐**. 수정 전후를 plain step과 대조해 확인 | 🔴 조용함 |

정상이었던 것: census 안에서 사용자 예외 전파, 활성값 체크포인팅과 census, bool·complex·float8·int8 버퍼 census, 없는 볼륨의 방출 위치로 doctor 실행.

**측정 방법 교정**: S7의 첫 판정에는 `p − lr·grad`를 기대값으로 썼는데, SGD의 결합 연산과 반올림이 달라 틀렸다. 이후 **같은 초기값의 plain step 결과와 비트 단위로 대조**하는 방식으로 바꿨다. 수정 전 동작(가드 비활성화)이 실제로 스텝을 건너뛴다는 것도 이 방식으로 다시 확인했다.

## 2차: 아키텍처·체크포인트·설정 (17종 → 결함 4건 + 적용 범위 문제 1건)

| # | 시험 | 결과 |
|---|---|---|
| R1 | LLaMA·BERT·T5·ViT 축소판 `from_pretrained` → 동면 → 추론 | 4종 모두 비트 동일. **ViT는 transformers 5가 불러올 때 가중치 이름을 바꿔**(`layers.0.attention.q_proj` ↔ 파일 `encoder.layer.0.attention.attention.query`) `source` 대응에 실패했다. 틀린 결과는 아니고 적용 범위가 줄었다 |
| R2 | 여러 파일로 나뉜 safetensors(8개) | ✅ 전부 `source` |
| R3 | 동면 중 HF `save_pretrained` | ✅(S1a 수정 후) 체크포인트 무결 |
| R4~R10 | 파라미터 개수 세기, inference_mode, 100회 반복(RSS 증가 4.9MiB, 살아 있는 핸들 1개), 핸들을 버린 뒤 `wake(tensor)`, 자식·부모 순서 동면, 동면 중인 모델의 census | ✅ |
| R11 | `configure(budget=[1,2])`, `idle_cells="abc"`, `headroom="x"`, `min_free_disk_fraction=None` | **결함 4건**: `ConfigError`가 아니라 원래 `TypeError`·`ValueError`가 새어 나옴 |

## 3차: 동시성·손상·환경 (8종 → 결함 1건)

| # | 시험 | 결과 |
|---|---|---|
| T1 | 6개 스레드가 같은 모델을 동시에 `now()` | **결함**: 다른 스레드가 이미 비운 텐서를 다시 압축 → 복원 불가 |
| T2~T8 | 동면 중 방출 파일 삭제, 작은 텐서 6,000개(동면 0.25초, 복원 0.07초), MPS에서 저장·deepcopy, torch import 실패 시 doctor, 오류 셀 뒤 노트북 추적기, 이중 wake와 GC 후 status, 모델과 Adam을 모두 동면한 학습 | ✅ |

## 수정

| 수정 | 해결한 결함 | 구현 |
|---|---|---|
| **가드**(`hibernate/_guards.py`): 동면 중인 객체에 닿는 모든 경로에서 먼저 깨운다 | S1a~e, S2, S7 | 모듈: forward·`state_dict`·`load_state_dict` pre-hook. 인스턴스 단위 `_apply`·`parameters`·`named_parameters`·`buffers`·`named_buffers`·`__reduce_ex__`·`__deepcopy__`. 옵티마이저: step·`state_dict`·`load_state_dict` pre-hook, pickle·deepcopy. 동면 중인 모든 텐서: `__reduce_ex__`·`__deepcopy__`, 그래디언트 hook. **프로세스 공용 optimizer step pre-hook**: 동면 중인 파라미터나 그래디언트를 가진 핸들을 깨운다. 동면 중인 텐서가 없으면 해제한다. 인스턴스 덮어쓰기는 깨울 때 **위임하기 전에** 지워서 pickle에 섞이지 않게 했다 |
| 복원 순서와 실패 처리 | 복원 실패 시 조용한 오류 방지 | 데이터를 먼저 복원하고 가드는 나중에 뗀다. **복원에 실패하면 동면 상태와 가드를 유지**한다(쓰려고 하면 계속 `IntegrityError`). `Handle.discard()`로 명시적으로 포기할 수 있다. 실패 메시지는 원인별로 요약한다 |
| 롤백 | S3 | `now()` 도중 `BaseException`(Ctrl-C 포함)이 나면 이미 비운 텐서를 모두 복원하고 다시 던진다 |
| 수명 | S4a·b | 레지스트리를 `WeakValueDictionary`로 바꿨다. 동면 중에는 객체가 가드를 통해 핸들을 붙잡으므로, 객체를 지우면 둘 다 수거된다. 방출 파일은 텐서가 사라질 때도 지운다(`weakref.finalize`) |
| 잠금 | S5, T1 | 핸들별 `RLock`(복원), 프로세스 공용 잠금(`now()` 직렬화). 이미 비워진 텐서는 건너뛴다 |
| 이름 변환 대응 | R1 ViT | transformers의 `revert_weight_conversion`으로 실행 중 텐서 → 파일 키를 대응시킨다. 같은 텐서 객체를 돌려주는 순수 이름 변경만 쓴다. 실패하거나 버전이 달라도 매칭이 줄어들 뿐이고, 모든 매칭은 비트 비교로 확인한다(K6) |
| 깨어 있는 이유 | 사용성 | 시도한 방법마다 이유를 모두 적는다(예: `source: 원본 없음; compress: 156%로 커짐; spill: 동의 필요`) |
| 설정 검증 | R11 | `TypeError`·`ValueError`도 `ConfigError`로 바꾸고 키와 값을 적는다 |
| census 종료 | 예외 가림 방지 | 사용자 예외가 있으면 census 실패로 가리지 않는다. 예외가 없으면 memopro 오류로 알린다 |
| `plan()` | 부작용 방지 | 동면 중인 객체에는 `InvalidArgument`(계획은 동면 전에 세우는 것). 부르면 깨어나 버리므로 막았다 |
| 노트북 대리 객체 | S1(노트북) | pickle·copy·deepcopy는 깨운 실제 텐서로 한다 |

## 관찰 (결함 아님): CPU에서 메모리 정렬과 계산 결과

실제 GPT-2(HF 캐시, CPU)에서 동면 → 복원 후 **파라미터 값은 전부 비트 단위로 같았다.** 그런데 logits는 최대 1.2e-4 달랐다. 원인을 분리한 결과는 다음과 같다.
- `from_pretrained`가 safetensors를 메모리 매핑하면 가중치 주소가 **64바이트 경계에서 19바이트 어긋나** 있다.
- **memopro 없이 값만 같은 `.clone()`으로 바꿔도 똑같이 1.2e-4가 달라진다.** 그 뒤로는 안정적이다.
- 즉 CPU BLAS(Accelerate)가 정렬 상태에 따라 다른 누적 순서를 쓰는 것이며, memopro와는 무관하다.
- memopro가 보장하는 것은 **텐서 값의 비트 동일**이다. 원래 정렬되지 않은 매핑 가중치는 어떤 재할당 뒤에든 첫 CPU 계산 결과의 마지막 자릿수가 바뀔 수 있다.
- 이전 시연(MPS·CUDA)은 장치로 복사되면서 정렬이 같아져 결과도 같았다. README 한계 절에 적는다.

## 검증

| 항목 | 결과 |
|---|---|
| 탐색 스크립트 3개 재실행 | 결함 **0/15, 0/17, 0/8** |
| 회귀 테스트 | `tests/test_hibernate_guards.py` **18개 추가**: 저장·복사·불러오기·개수·dtype, optimizer step(plain step과 비트 대조), 모델+Adam 학습 동일성, 중단 롤백, 스레드 호출·동시 동면, 수명, 핸들 없이 복원, 파일 손실 후 가드 유지와 discard, ViT 이름 변환, 여러 파일 체크포인트, 이유 목록, 설정 타입, census 예외 |
| 전체 | `pytest` **123 통과**(105 → 123), `cargo test` 21 통과, ruff·clippy 통과 |
| 실제 GPT-2 | 전부 `source`, 깨어 있는 텐서 0개, 파라미터 비트 동일 |

## 한계

- 가드는 **torch의 공개 hook API와 인스턴스 속성**에 기댄다. torch 내부 경로가 이 속성을 거치지 않고 `module._parameters`를 직접 읽으면 잡지 못한다(예: 일부 외부 라이브러리의 직접 접근). 이 경우 텐서는 크기 0이라 대개 형상 오류로 크게 실패한다.
- 핸들 API로 동면시킨 **일반 텐서를 직접 연산**하면(`t + 1`) 깨우지 않는다. 크기 0이라 오류가 나거나, 브로드캐스트되는 연산에서는 빈 결과가 나올 수 있다. 문서에 "`h.wake()` 전에는 쓰지 말 것"이라고 적었다. 노트북 대리 객체는 자동으로 깨운다.
- 프로세스 공용 optimizer hook은 동면 중인 파라미터가 있을 때만 등록된다(P4의 전역 변경 최소화).
- `revert_weight_conversion`은 transformers 5의 내부 모듈이다. 버전이 바뀌면 이름 변환 모델의 `source` 적용 범위만 줄어든다.

## 논문 매핑

- **논문 B System**: 가드 설계("동면 중인 객체에 닿는 모든 경로에서 먼저 깨운다")와 복원 실패 시 가드 유지.
- **논문 B Threats to Validity**: 비트 동일의 범위(텐서 값)와 CPU BLAS의 정렬 의존성, hook이 닿지 않는 직접 접근.
- **연구 방법론**: 탐색적 시험 40종으로 결함 16건을 찾았고, 그중 조용한 오류가 6건이었다. 판정 방법 자체의 오류(S7 기대값)를 교정한 사례도 포함한다.
