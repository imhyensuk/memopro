# 0096. E022 사전 등록: 0094 수정(D1~D4)의 Colab T4 재측정 — 통합 셀 하나

- **날짜**: 2026-09-29
- **유형**: experiment (사전 등록) + tool (노트북)
- **상태**: 확정. 실행은 사용자가 Colab에서 한다. 결과는 다음 기록
- **관련 기록**: 0093 (E021 결과와 결함), 0094 (D1~D4 수정), 0091·0092 (노트북 구조)

## 요청

> "Colab 재측정용 노트북 파일 보내줘. 이전처럼 3개가 아니라 통합 파일 셀로 만들어줘."

## 1. 노트북 (`examples/colab_t4_remeasure.ipynb`, 원본 `examples/colab_t4/cell_remeasure.py`)

- **통합 코드 셀 하나**에 0094 수정의 효과를 보는 경우만 담았다. 7차의 공통 코드·워커를 그대로 쓴다.
- **경우** (모두 새 프로세스)
  - 회귀 검사 7종. `platform_defaults`는 "run의 γ 꺼짐"을 기대한다(0094).
  - 추론(7차와 같은 측정: 불러오기, 첫 토큰 512·2048, 128토큰 디코딩, PPL 8×1024, 답변)
    - 1.5B·3B: HF fp16 auto, memopro.
    - 7B: HF bnb4·bnb8, memopro(기본), memopro low.
  - 7B QLoRA 20단계: 표준 방식 대 memopro.
  - 수정하지 않은 스크립트를 `memopro run`으로: 1.5B·7B.
- **요약** `summary.md` 맨 위에 아래 기준의 **pass / fail / n/a 판정 표**를 자동으로 계산해 넣는다.
- **공통 코드 개선** (7차 노트북에도 적용)
  - 셀의 진행 로그를 결과 폴더의 `orchestrator.log`에 저장한다.
  - 모델을 로컬 디스크로 복사하지 않을 때 크기와 여유를 남긴다(0093의 14B 사례).
  - `env.json`의 `build_has`에 `0094_fixes`를 더했다(fp16 우선 문구가 있고 `run`의 γ 기본이 꺼짐).
- **설치**: 0094가 들어간 새 sdist(main `ecd604d`의 라이브러리 코드)를 Drive `memopro_colab/install/`에 **바꿔 놓아야** 한다. 이전 세션에서 설치했다면 런타임을 다시 시작한다.
- 로컬 MPS에서 작은 설정(GPT-2)으로 끝까지 돌려 판정 표와 `orchestrator.log`가 만들어짐을 확인했다. CUDA가 없어 판정은 n/a·fail로 나왔고, 측정에는 쓰지 않는다.

## 2. 판정 기준 (실행 전 고정)

| # | 기준 |
|---|---|
| R1 | 회귀 검사 7종 모두 통과 |
| D1 | memopro QLoRA가 완주하고, `train_session.plan` 보고가 모두 `applied`다(`skipped` 없음) |
| D2 | 1.5B·3B 각각: memopro가 `load.half`(fp16)를 고르고, 2048토큰 첫 토큰 시간 ≤ HF fp16 auto의 1.2배 |
| D3 low | 7B `quality="low"`가 `load.quant.int4`를 고르고, 디코딩 ≥ HF bnb4의 0.9배 |
| D3 default | 7B 기본 품질은 계속 `load.quant.int8`을 고른다 |
| D4 7B | `memopro run`이 로딩 정책을 적용하고("does not fit as stored: loaded as …"), γ 기록(`[applied] elastic`)이 없다 |
| D4 1.5B | `memopro run`이 "fits as stored"로 개입하지 않는다(D2가 있어도) |

- **보고만 하는 항목**
  - D2의 디코딩 속도와 PPL(bf16 → fp16 변화).
  - QLoRA 속도(memopro int4 대 표준 nf4).
  - 7차 대비 변화.
- **결론 규칙**
  - 어느 기준이든 실패하면 그 결함을 다시 연다(원인 기록 후 수정).
  - D2 PPL이 HF fp16과 1% 넘게 다르면 fp16 넘침을 의심해 조사한다.

## 한계 (미리 적어 둠)

- 7차와 같은 T4 한 종, 같은 모델이다. 세션마다 Colab 기기 상태가 다를 수 있어, 7차 수치와의 직접 비교는 참고로만 둔다. 판정은 같은 실행 안의 비교로 한다.
- 7B QLoRA의 memopro 쪽은 D3 때문에 이제 int4를 쓴다. 표준 방식(nf4 + 이중 양자화)과는 여전히 설정이 조금 다르다.

## 논문 매핑

- **논문 B Evaluation**: 결함 → 수정 → 재측정의 폐회로. 판정 표를 그대로 싣는다.
