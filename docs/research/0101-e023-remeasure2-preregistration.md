# 0101. E023 사전 등록: 0099·0100 수정의 Colab T4 재측정 — 통합 셀 하나

- **날짜**: 2026-09-29
- **유형**: experiment (사전 등록) + tool (노트북)
- **상태**: 확정. 실행은 사용자가 Colab에서 한다. 결과는 다음 기록
- **관련 기록**: 0099 (D8+D5, D9, O1), 0100 (정밀 검토), 0098 (E022), 0093 (E021: GPT-2·0.5B 학습 기준값)

## 요청

> "PR #28 병합하고 재측정용 노트북 파일 보내줘. 파일 작성 전에 다시 한 번 정밀하게 코드들을 분석해서 버그나 오류 포인트를 보완해."

## 1. 노트북 (`examples/colab_t4_remeasure2.ipynb`, 원본 `examples/colab_t4/cell_remeasure2.py`)

- **통합 코드 셀 하나**. 공통 코드와 워커는 7차·E022와 같고, 0100의 보완(원자적 복사, 모델 하나만 남기기, 빌드 확인 후 중단)이 들어갔다.
- **경우** (모두 새 프로세스, 학습 설정은 7차와 같음: fp32 AdamW, lr 1e-5, WikiText-2, 같은 시드·데이터, dropout 끔)

| 순서 | 경우 | 목적 |
|---|---|---|
| 1 | 회귀 검사 7종(Qwen2.5-0.5B-Instruct) | R1, D9 |
| 2 | Qwen2.5-0.5B 배치 8·512토큰 30단계: memopro, memopro micro 1 고정 | T3 |
| 3 | GPT-2 배치 16·512토큰 30단계: HF checkpointing(정확한 전체 배치 기준), memopro, memopro 30% 상한 | T1, T2 |
| 4 | Qwen2.5-7B-Instruct: memopro `load` 기본 품질(추론 측정 7차와 같음), `memopro run`(수정하지 않은 스크립트), QLoRA 20단계 표준 방식 대 memopro | O1, T4 |

- 7B는 로컬 디스크에 한 번만 복사한다. 예상 시간은 45~60분이다(sdist 빌드 3~5분 포함).
- **설치**: 0099·0100이 들어간 새 sdist(이 기록의 커밋)를 Drive `memopro_colab/install/`에 **바꿔 놓는다**. 이전 세션에서 설치했다면 런타임을 다시 시작한다. 빌드에 수정이 없으면 셀이 측정 전에 멈춘다.

## 2. 판정 기준 (실행 전 고정)

| # | 기준 |
|---|---|
| R1 | 회귀 검사 7종 모두 통과 |
| D9 | CUDA int4 불러오기의 보고에 bitsandbytes 품질 문구(`BNB_INT4_QUALITY_NOTE`)가 있다 |
| T1 | GPT-2 전체 T4: 완주, 재시도 0, micro-batch가 고른 분할(micro = ⌈n/⌈n/micro⌉⌉), 단계별 손실의 최대 상대 차이 ≤ 1e-5 (HF checkpointing 대비) |
| T2 | GPT-2 30% 상한: 완주, OOM 재시도 0, 손실 최대 상대 차이 ≤ 1e-5 (HF checkpointing 전체 T4 대비) |
| T3 | 0.5B 전체 T4: 완주, micro ≥ 2, 재시도 0, 손실 최대 상대 차이 ≤ 1e-5 (micro 1 고정 대비) |
| T4 | 7B QLoRA: 완주, micro ≥ 2, 재시도 0, 속도 ≥ 표준 방식의 0.95배, 손실 최대 상대 차이 ≤ 5e-3 (양자화 설정이 조금 다름: 이중 양자화) |
| O1 load | 7B 기본 품질이 여전히 `load.quant.int8`을 고르고, `suggested` 보고에 "would load int4"와 "2.9x"가 있다 |
| O1 run | `memopro run` 7B가 "does not fit as stored"로 정책을 적용하고, 보고에 "memopro run --quality low would load int4"가 있다 |

- **계산으로 한 예측**(판정 기준이 아님, 결과 해석용. 0099의 규칙과 7차·E022 수치로 계산)
  - T1 micro 8(8+8).
  - T2 micro 2. 7차는 3에서 OOM 뒤 2였다.
  - T3 micro 2. 7차는 1이었다.
  - T4 micro 4. 세션 시작 때 캐시 여유가 없으면 2(2+2)다.
- **보고만 하는 항목**
  - tok/s와 7차·E022 대비 변화: GPT-2 3632(micro 15), 0.5B 889(micro 1), QLoRA 194 대 221.
  - 최대 할당·예약 메모리.
  - 세션 시작 때의 free·캐시 여유(`session_start`).
  - 7B 기본 디코딩 속도.
- **결론 규칙**
  - T1~T4 가운데 하나라도 실패하면 그 원인을 기록하고 계획 규칙(여유 계수, 캐시 포함)을 다시 연다. 기준은 바꾸지 않는다.
  - OOM 재시도가 생기면 여유 계수 1.25가 부족하다는 뜻으로 본다.

## 한계 (미리 적어 둠)

- 7차·E022와 같은 T4 한 종, 같은 모델 계열이다. 세션마다 기기 상태가 달라 tok/s의 실행 간 비교는 참고로만 둔다. 판정은 같은 실행 안의 비교로 한다.
- T3의 정확성 기준은 memopro 자신(micro 1 고정)이다. 전체 배치 기준(plain, HF ckpt)은 0.5B에서 OOM이라 쓸 수 없다(7차).
- 30% 상한은 `set_per_process_memory_fraction`으로 흉내 낸 것이다. 실제 작은 GPU와 드라이버·컨텍스트 몫이 다르다.

## 논문 매핑

- **논문 B Evaluation**: 결함(D8·D5) → 수정(0099·0100) → 사전 등록 재측정. 계획의 크기·정확성·속도를 한 표에 싣는다.
