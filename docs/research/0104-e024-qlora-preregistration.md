# 0104. E024 사전 등록: 0103(여유 계수 1.3, 캐시 실측)의 7B QLoRA 재측정 — 통합 셀 하나

- **날짜**: 2026-09-29
- **유형**: experiment (사전 등록) + tool (노트북)
- **상태**: 확정. 실행은 사용자가 Colab에서 한다. 결과는 다음 기록
- **관련 기록**: 0103 (P-a·P-b), 0102 (E023 F1·F2), 0101 (E023 사전 등록)

## 요청

> "PR #31 병합하고 QLoRA 재측정 노트북 보내줘"

## 1. 노트북 (`examples/colab_t4_qlora.ipynb`, 원본 `examples/colab_t4/cell_qlora.py`)

- **통합 코드 셀 하나**. 공통 코드와 워커는 E023과 같다.
- **경우** (새 프로세스, E022·E023과 같은 설정): Qwen2.5-7B-Instruct QLoRA, 배치 4·512토큰, 20단계, lr 2e-4, LoRA r=16(7개 투영), gradient checkpointing.
  - 표준 방식: bnb nf4 + 이중 양자화 + fp16 계산 + peft.
  - memopro: `load(quality="low")` + peft + `train_session`.
- `env.json`의 `build_has`에 `0103_fixes`를 더했다. 없으면 셀이 측정 전에 멈춘다.
- **요약**
  - 판정 표.
  - 학습 시작 전(`session_start`)과 memopro가 잰 직후(`session_measured`)의 free·allocated·reserved·캐시.
  - memopro가 돌려준 캐시 양과 계획 문구.
- 예상 시간은 20~25분이다(sdist 빌드와 7B 로컬 복사 포함).
- **설치**: 0103이 들어간 새 sdist를 Drive `memopro_colab/install/`에 **바꿔 놓고**, 이전 세션에서 설치했다면 런타임을 다시 시작한다.
- 로컬 MPS에서 셀 로직을 끝까지 돌려 요약이 만들어짐을 확인했다. QLoRA는 CUDA 전용이라 두 경우 모두 건너뛰었고 판정은 n/a였다.

## 2. 판정 기준 (실행 전 고정)

| # | 기준 |
|---|---|
| Q1 | memopro가 20단계를 완주하고 OOM 재시도가 0이다 |
| Q2 | memopro의 최대 예약 메모리 ≤ memopro 계획 보고의 예산(`budget …`). E023에서는 13.90 > 13.75GiB였다 |
| Q3 | memopro 속도 ≥ 표준 방식의 0.9배, 단계별 손실의 최대 상대 차이 ≤ 5e-3 |

- **보고만 하는 항목**
  - memopro가 돌려준 캐시와 남은 캐시(조각).
  - 고른 micro. 0103의 예상은 4, 조각이 많으면 2+2다.
  - 최대 할당·예약·GPU 사용.
- **결론 규칙**
  - Q2가 실패하면 캐시 조각 외의 원인(여유 계수, 불러오기 뒤 남은 할당)을 기록하고 다시 연다.
  - Q3이 실패하고 micro가 2라면, 안전과 속도의 교환으로 기록하고 사용자에게 묻는다.

## 한계

- 한 기기(T4), 한 모델, 한 번의 실행이다. 기기 상태에 따른 속도 편차는 E023(1.00배)과의 비교에서 참고로만 둔다.

## 논문 매핑

- **논문 B Evaluation**: 캐시 실측 전환(P-b)이 예산 준수(Q2)와 속도(Q3)에 주는 영향.
