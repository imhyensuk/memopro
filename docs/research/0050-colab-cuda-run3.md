# 0050. Colab T4 3차 실행: 가드와 torch.compile을 CUDA에서 확인

- **날짜**: 2026-09-26
- **유형**: experiment (사전 등록한 검사의 실행)
- **상태**: 확정
- **관련 기록**: 0049 (사전 등록, torch.compile 수정), 0048 (가드), 0045·0046 (Colab 1·2차)

## 배경

0049에서 사전 등록한 3차 검사를 사용자가 Google Colab T4에서 실행했다. 판정 대상은 세 구역이다.
- `guards_cuda` 14개, `guards_gpt2` 4개, `guards_compile` 7개. **모두 `"ok": true`여야 통과**다.

1·2차 구역(doctor, census_cuda, hibernate_gpt2, backward_in_flight)은 새 코드로 다시 돌린 회귀 확인이다.

## 실험 환경

| 항목 | 값 |
|---|---|
| 하드웨어 | Google Colab, NVIDIA Tesla T4 (여유 15,413,870,592바이트) |
| PyTorch / transformers | 2.11.0+cu128 / 5.16.1 (Python 버전은 기록하지 않음) |
| memopro | 0.1.0a1, 커밋 `7a773eb`에서 로컬 교차 빌드한 wheel (SHA-256 `c484ad31…`, 0049). 노트북 첫 출력 `guards_0049: true`로 수정 포함을 확인 |
| 절차 | wheel을 `/content`에 올리고 **Runtime → Run all**, 마지막 셀의 JSON을 그대로 붙여 넣음 |
| 원시 결과 | `docs/research/data/v2_colab/colab_t4_run3.json` |

## 결과: 판정 25/25 통과

| 구역 | 결과 | 주요 수치 |
|---|---|---|
| `guards_cuda` | **14/14** | 저장·복사·불러오기·이동·개수·스텝(foreach, fused)·학습·중단·스레드·수명·정리 모두 통과. 학습에서 동면 1회당 CUDA 840,192바이트 해제. 이 값은 모델과 AdamW 상태의 논리 크기 838,848바이트를 할당 단위 512바이트로 올린 값과 정확히 같다. 학습 통제 실행은 결정적이었다 |
| `guards_gpt2` | **4/4** | 동면 중 `generate`(48토큰), `state_dict`, `save_pretrained` 후 다시 불러오기, backward와 step 사이 동면(파라미터 `source` 497,759,232 + 그래디언트 `host` 497,759,232)이 모두 비트 동일. 첫 동면에 CUDA 498,690,048바이트 해제 |
| `guards_compile` | **7/7** | inductor·triton(T4), `fullgraph=True`, `m.compile()`, `mode="reduce-overhead"`(**CUDA graphs**): 컴파일 출력과 비트 동일, 분할 0, 새 그래프 0, 경고 0. 동면 뒤 컴파일: 값이 맞고 경고 1회. 컴파일된 학습+동면이 동면 없는 학습과 비트 동일(통제 결정적). `torch.compile(opt.step)` 정상 |

회귀 확인(1·2차 구역)도 모두 이전과 같다.
- doctor: 보고한 여유 = torch 값(15,413,870,592).
- hibernate_gpt2: `host`·`auto`(전부 `source`) 모두 비트 동일. CUDA 예약 메모리 532.7MB·534.8MB 해제.
- backward_in_flight: 그래디언트가 같다. 공유 텐서 2개는 깨어 있는 채로 이유가 보고됐다(2차와 같다).

**로컬 교차 빌드한 wheel**(zig, manylinux2014)도 Colab에서 설치·실행됐다. 0049의 빌드 방법이 동작함을 확인한 것이다.

## 관찰 (판정 대상 아님)

**O1. census 적용률 100% → 98.9%.** 미분류 CUDA 메모리가 2,048바이트에서 1,126,912바이트로 늘었다.
- cuBLAS 작업공간은 2차와 같았다(18,087,936 = 독립 측정).
- 2차 wheel(`8d44d70`) 이후 census 코드는 예외 처리만 바뀌었다(`git diff`로 확인). 집계 방식은 그대로다.
- 대신 저장된 활성값이 32,379,460 → 30,284,356바이트(−2,095,104)로 줄었다. 즉 **모델의 forward 코드 자체가 달라졌다**. Colab 환경(transformers 등)이 바뀐 것이다. 2차는 transformers 버전을 기록하지 않아 직접 비교할 수 없다.
- 로컬(transformers 5.17, CPU)의 같은 GPT-2 모듈은 파라미터·버퍼가 아닌 텐서를 들고 있지 않았다.
- **원인은 찾지 못했다.** 가설: CUDA에서만 생기는 transformers나 SDPA의 할당.
- census는 이 메모리를 숨기지 않고 "미분류"로 보고한다(설계대로). 그러나 0046의 "적용률 100%"는 **환경에 따라 달라지는 값**이다. 논문에는 환경별 적용률로 보고해야 한다.
- 다음 CUDA 실행에서 `torch.cuda.memory_snapshot()`으로 해당 블록을 식별한다.

**O2. `host`의 RSS 증가량이 2차와 다르다.** 같은 497,759,232바이트를 호스트로 옮겼는데 측정한 RSS 변화는 다음과 같았다.
- 2차: −494,833,664 / 3차: −151,457,792
- 할당자가 이미 상주 중인 해제된 페이지를 재사용하면 RSS는 덜 늘어난다. 그래서 측정값은 "새로 쓴 RAM"이지 논리 크기가 아니다.
- 부호 있는 보고(0045 F1)는 의도대로 음수(추가)를 보였다. README 한계 절("회수량은 실측")이 이미 다룬다.
- 논문에는 논리 바이트와 측정 RSS를 따로 보고한다.

**O3.** `source`의 임시 RAM(−18.9MB)은 2차와 같다. 알려진 개선 과제다.

## 결론

- 0048의 가드와 0049의 torch.compile 수정이 **실제 NVIDIA GPU에서도** 사전 등록한 조건을 모두 만족했다.
- CPU에서 확인할 수 없던 경로도 포함한다: `host`, fused AdamW, triton, CUDA graphs.
- 출시 전 관문 V2(CUDA)는 가드와 컴파일까지 넓혀 통과했다.

## 한계 및 향후 과제

- GPU 1종(T4, sm75), 실행 1회, torch 2.11만 시험했다.
- 시험하지 않은 것: bf16이 기본인 GPU(A100·H100)와 flash attention, DDP·FSDP, compiled autograd.
- Python 버전을 기록하지 않았다. 다음 노트북은 기록한다.
- O1의 원인 식별(메모리 스냅샷)과 `source` 임시 RAM 축소가 남아 있다.

## 논문 매핑

- **논문 B Evaluation**: 가드 25종(저장·복사·이동·스텝·학습·중단·스레드·수명·컴파일)을 CUDA에서 사전 등록한 기준으로 판정. CUDA graphs를 포함한 컴파일 경로에서 비트 동일, 재컴파일 0.
- **논문 A·B Threats to Validity**: census 적용률이 환경(라이브러리 버전)에 따라 98.9~100%로 달라진다. 측정 RSS는 할당자 재사용 때문에 논리 크기와 다르다.
