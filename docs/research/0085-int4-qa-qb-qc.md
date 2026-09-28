# 0085. Q-a~Q-c 구현: int4 품질 비용 보고, 그룹 32 기본값, 긴 입력은 풀어서 bf16 행렬곱(1024토큰 프롬프트 29.2 → 8.7초)

- **날짜**: 2026-09-29
- **유형**: implementation (사용자 결정) + experiment (적용 전 확인)
- **상태**: 확정
- **관련 기록**: 0084 (E020 결과와 제안 Q-a~Q-c), 0069 (int4pack), 0072 (파일 캐시)

## 사용자 결정

> "PR #17 병합하고 Q-a~Q-c 적용해"

## 구현

- **Q-a** (`orchestrator/candidates.py::quality_note`, `access/_load.py`)
  - int4로 불러오면(일반 경로, 파일 기반 경로 모두) `applied` 보고 뒤에 품질 비용을 붙인다: "quality cost: int4 raised WikiText-2 perplexity by 5.7-7.6% for Qwen2.5-1.5B/3B with torch int4pack group 32 (E020, 0084); other models and back ends differ".
- **Q-b** (`int4pack.GROUP = 32`)
  - 크기 추정 `ModelInfo.weight_bytes(bits=4, group=…)`에 그룹 인자를 더했다. torch int4pack이면 32, 다른 백엔드면 64다(`candidates._int4_group`).
  - 파일 캐시 키에 그룹이 들어 있으므로 새 캐시를 만든다. 옛 g64 캐시는 디스크에 남는다(자동 삭제하지 않음).
- **Q-c** (`Int4PackedLinear.dequantized()`, `forward`, `int4pack.LONG_INPUT = 160`)
  - 입력 행이 `LONG_INPUT` 이상이면 그 층의 int4를 bf16으로 풀어 `F.linear`로 계산하고, 임시 가중치는 버린다. 그보다 짧으면 기존 커널을 쓴다.
  - **묶음 형식**: MPS의 `_convert_weight_to_int4pack` 결과는 겉모양이 `[N/8, K/128, 32, 4]` int32지만, 실제 메모리는 `[N, K]` 코드를 int32마다 8개(낮은 니블부터) 행 순서로 담은 것이다.
    - 모양 세 가지(16×256, 32×512, 24×384)에서 탐침 행렬로 확인했다.
    - 그래서 모양 바꾸기와 비트 연산만으로 코드를 복원한다.
  - **역양자화 공식**: torchao tinygemm 관례 `w = (q − 8) × scale + zero`를 float32로 계산한 뒤 bf16으로 바꾼다.
  - 새 커널이나 양자화 방식은 없다(0010). torch 연산만 쓴다.

## 적용 전 확인 (`experiments/e020_int4_quality/followup.py`, 결과 `data/e020/followup_*.json`)

- **복원 정확도**
  - torchao 역양자화와의 차이는 최대 원소 크기의 bf16 한 단계 이내였다(모양 4종).
  - 커널 출력과의 상대 차이는 0.3~0.5%로, bf16 행렬곱의 누적 차이 수준이다.
- **디코딩 속도** (16토큰 프롬프트 뒤 64토큰, 3회 중앙값, 그룹을 번갈아 2번씩 측정)

| 모델 | g64 | g32 | 차이 |
|---|---|---|---|
| 1.5B | 37.1 / 37.1 tok/s | 35.6 / 35.7 tok/s | −4% |
| 3B (다시 잰 값) | 11.8 / 11.9 tok/s | 11.6 / 11.7 tok/s | −2% |

- **3B 첫 측정은 오염됐다.**
  - 3B 첫 측정(g64 19.0, g32 19.7 → 11.3 tok/s)은 같은 시간에 제가 MPS 테스트(`pytest tests/test_int4pack.py`)를 함께 돌려 섞였다.
  - 그래서 3B만 다른 작업 없이 다시 쟀다. 오염된 결과는 `data/e020/followup_contaminated/`에 따로 남겼다.
  - 다시 잰 값(11.6~11.9)과 첫 값(19~20)의 차이는 기기 상태(메모리)에 따른 것으로 보인다. 그룹 사이의 비교는 같은 실행 안에서 번갈아 잰 값으로만 한다.
- **전환 지점** (순전파 한 번, 커널만 대 풀어서 계산만)
  - 1.5B g32: 128행 2.19 대 3.04초, 192행 3.26 대 3.16초 → 약 180행.
  - 3B g32: 128행 8.1 대 8.8초, 192행 12.2 대 9.3초 → 약 150행.
  - 그래서 `LONG_INPUT = 160`으로 정했다.
  - 1024행에서는 1.5B 18.7 대 5.6초, 3B 65.6 대 17.0초였다.
- **끝에서 끝까지** (`memopro.load(Qwen2.5-1.5B, quality="low")`, `generate`)
  - 1024토큰 프롬프트와 8토큰 생성: 혼합 경로 **8.72초**, 커널만 29.19초. 생성 토큰은 같았다.
  - 보고: "quant.int4 via torch-int4pack …, weights 1.20 GiB on mps; quality cost: …".

## 시험

- `tests/test_int4pack.py` 5개를 추가했다(모두 12개).
  - 기본 그룹 32.
  - 복원이 torchao와 같음(모양 4종).
  - 긴 입력은 커널을 부르지 않고 결과가 같음.
  - 보고에 품질 비용 문구가 들어감.
  - 크기 추정에 그룹 인자가 있음.
- 전체 Python 246개 통과. ruff 통과.

## 한계

- g32의 품질 수치(E020)는 WikiText-2 앞 22%, Qwen2.5 두 모델 기준이다.
- 전환 지점은 이 기기(M1 8GB)와 두 모델에서만 쟀다. 역양자화 고정 비용은 기기 상태에 따라 1.5~2배 달라졌다(3B 5.2~7.8초).
- 긴 입력 경로는 층마다 임시 bf16 가중치(최대 약 45MB)와 float32 중간값(최대 약 90MB)을 만든다. 순간 메모리가 조금 는다.
- 기존 g64 파일 캐시(E017에서 만든 것)는 이제 쓰이지 않는다. `residency="file"`로 int4를 쓰려면 캐시를 다시 만들어야 한다(`disk_writes="allow"`).

## 논문 매핑

- **논문 C Evaluation**: 긴 프롬프트 첫 응답 시간(1024토큰 29.2 → 8.7초)과 그룹 32의 품질·속도 교환(PPL 손실 약 1/3, 디코딩 −2~4%).
- **논문 B**: 기존 기법(torch 커널, torchao 양자화)의 조합만으로 사용성 문제를 푼 사례. 보고에 품질 비용을 적는 설계.
