# 0069. MPS int4 백엔드 교체: bitsandbytes → torch `_weight_int4pack_mm`

- **날짜**: 2026-09-28
- **유형**: decision + milestone (구현)
- **상태**: 확정
- **관련 기록**: 0068 (E015 Q4: torch int4pack 1.63배, bitsandbytes nf4 0.35배), 0063 (E014 D5: int4 4.9 tok/s), 0052 E3 (양자화 백엔드는 실제로 돌아야 사용)

## 배경

> "PR #10 병합하고 int4 백엔드 교체 후 E016 진행해"

- E014에서 MPS의 1.5B int4는 **4.9 tok/s**로 bf16(11 tok/s)보다 느렸다.
- E015 Q4에서 원인을 찾았다. 백엔드 순서가 bitsandbytes → torchao인데, bitsandbytes nf4는 MPS에서 bf16 행렬곱의 0.35배 속도다.
- torch에 기본으로 들어 있는 `_weight_int4pack_mm`(MPS)은 1.63배다.
- torchao의 int4(`Int4WeightOnlyConfig`)는 MPS에서 `mslk` 패키지를 요구하며 실패한다(이 기기에서 확인).

## 결정

1. **MPS의 int4 백엔드 순서**: `torch-int4pack` → bitsandbytes → torchao. CPU·CUDA는 바뀌지 않는다(`torch-int4pack`은 MPS에서만 쓴다).
2. **구현** (`techniques/integrations/int4pack.py`)
   - 커널은 torch의 `_weight_int4pack_mm`이다.
   - 양자화와 형식은 torchao의 `groupwise_affine_quantize_tensor`(tinygemm 형식, 그룹 64)다.
   - 두 결과의 정렬은 torch `_convert_weight_to_int4pack`(inner k tiles 8)이 맡는다.
   - memopro는 이들을 잇는 코드(`Int4PackedLinear`, `convert`)만 가진다. 0010의 "재구현하지 않는다"에 맞다.
3. **불러오는 경로**
   - `from_pretrained(device_map="cpu", dtype="auto")`로 원래 형식을 CPU에 올린다. safetensors가 mmap되므로 원본 bf16은 깨끗한 파일 페이지다.
   - 선형층을 하나씩 int4로 바꿔 장치에 올린 뒤, 나머지(임베딩, 헤드, 정규화)를 bf16으로 옮긴다.
   - 그래서 **bf16 전체가 장치에 올라가지 않는다**. 니블 묶기는 CPU에서 해서 장치의 임시 텐서를 줄인다.
4. **제외하는 층**: 임베딩·헤드·정규화(`_info`와 같은 이름 규칙), 그리고 커널이 받지 않는 모양(K가 128의 배수가 아닌 층)은 bf16으로 남긴다.
5. **보고와 연결**
   - `report()`에 `quant.int4 via torch-int4pack`을 적는다.
   - `memopro run`의 로딩 정책도 같은 후처리(`post_load`)를 쓴다.
   - 후보 구성의 크기 추정(`_info.weight_bytes(bits=4)`)은 바뀌지 않는다.

## 결과

**테스트** (`tests/test_int4pack.py`, MPS에서만 실행되고 CI VM에서는 건너뜀)
- 백엔드 순서.
- `Int4PackedLinear`가 역양자화 가중치의 행렬곱과 상대 오차 1% 안에서 일치한다.
- 작은 Qwen2의 14개 층을 변환하면 역양자화 기준 모델과 로짓 상대 오차 3% 안에서 일치한다. 헤드는 제외된다.
- `load`가 `torch-int4pack`을 고르고 생성한다.
- 전체 테스트는 Python 224개가 통과했다.

**실제 모델** (Qwen2.5-1.5B-Instruct, M1 8GB, `quality="low"`, `budget="2GB!"`)

| | E014 (bitsandbytes nf4) | 이번 (torch int4pack) | 참고: bf16 |
|---|---|---|---|
| 생성 속도(64토큰) | 4.9 tok/s | **14.7 tok/s** | 11.0 tok/s |
| 불러오기 | 9.4s | 10.4s | 4.1s(memopro) / 10.1s(`naive`) |
| 장치 가중치 | — | 1.12~1.16GiB | 2.88GiB |
| 프로세스 footprint | 1.94GB(최대) | 2.32GB(캐시 끔 2.19GB) | 3.5~3.6GB |

- 출력은 "A GPU can run out of memory if it does not have enough RAM to store all the data…"로 뜻이 통한다.
- **남은 비용**: MPS 드라이버 할당량(1.82GB)이 실제 텐서(1.16GB)보다 약 0.67GB 크다.
  - 변환 중 임시 텐서를 해제하고 `empty_cache`를 불렀는데도 남는다.
  - 그래픽 드라이버가 해제된 메모리를 붙잡는 현상(0062 §1)과 같은 종류로 보인다.
  - E016의 F 등급(파일 매핑 int4)은 이 변환 자체를 없앤다.

## 한계

- 양자화 품질(nf4 대 group-wise 비대칭 int4)은 비교하지 않았다. 두 방식 모두 품질 등급은 `moderate_loss`다.
- 속도는 한 번 잰 값이다. 사전 등록 비교는 E016에서 한다.

## 논문 매핑

- **논문 B System**: 백엔드 선택을 실측으로 정하는 원칙(0052 E3)이 3배의 속도 차이를 만든 사례.
- **논문 B Evaluation**: 1.5B int4의 생성 속도 4.9 → 14.7 tok/s.
