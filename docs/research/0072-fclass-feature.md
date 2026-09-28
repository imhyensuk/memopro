# 0072. RCR F 등급의 기능화: `residency="file"`

- **날짜**: 2026-09-28
- **유형**: decision + milestone (구현)
- **상태**: 확정
- **관련 기록**: 0071 (E016: rcr_int4가 1.5B·3B에서 C2 만족, footprint 0.53GB, 스왑 0), 0068 (G-F), 0066 (RCR), 0069 (int4 백엔드), 0059 (디스크 풀)

## 배경

> "PR #11 병합하고 F 등급 기능화 후 E017 진행해"

E016은 실험 코드(`experiments/e016_rcr/fclass.py`)였다. Objective-C 도우미를 실행할 때 clang으로 빌드해야 하고, 오류 처리·정책(디스크 동의, 예산)·보고가 없었다. 이 기록은 그것을 memopro 기능으로 만든다.

## 결정

1. **사용자 표면**
   - 새 설정 `residency`: `"memory"`(기본, 지금과 같음) 또는 `"file"`(F 등급).
   - `configure`, `using`, `load(..., residency="file")`, `memopro.toml`, `MEMOPRO_RESIDENCY` 어디서든 쓸 수 있다.
   - **구성 선택은 바뀌지 않는다.** 예산·품질·선호로 고른 구성을 **어떤 메모리 종류로 쥘지**만 바꾼다.
   - E016에서 bf16 3B가 여유를 넘자 붕괴했으므로, "여유 안에 들어가는 구성"을 고르는 기존 선택이 그대로 전제 조건 역할을 한다.
2. **지원 범위** (MPS, 통합 메모리)
   - `stored`: 원본 safetensors를 **그대로** 매핑한다(변환·추가 디스크 없음).
   - `half`, `quant.int4`(torch int4pack): 한 번 만든 **F 캐시 파일**을 매핑한다.
   - 그 밖(int8 백엔드의 텐서 하위 클래스, 오프로드)과 MPS가 아닌 장치에서는 `ModeUnavailable`을 이유·대안과 함께 낸다. 사용자가 명시한 방식이므로 조용히 대체하지 않는다.
3. **F 캐시 파일**
   - 위치: `spill_location()/fcache/<키>.f`와 `.json` 목록.
   - 키: 모델 출처, 리비전, 구성 이름, int4 그룹, torch 버전, 형식 버전의 해시.
   - 목록에는 원본 파일의 크기와 수정 시각을 적는다. 다르면 다시 만든다.
   - **만들 때는 SSD에 쓰므로 `disk_writes="allow"`가 필요하고, 디스크 풀 예산(0059)과 20% 하한을 지킨다.** 이미 있는 유효한 캐시를 읽는 데는 동의가 필요 없다.
   - 파일은 사용자만 읽을 수 있고(0600), 텐서마다 16KiB 정렬한다. 수동 삭제 전까지 남는다. 캐시이므로 종료 시 지우지 않는다(방출 파일과 다름).
4. **Rust 코어 (`residency.rs`, 새 크레이트 없음)**
   - `FileMap`: 읽기 전용 `mmap`(페이지 단위로 올림), `mincore` 상주 비율, `pread` 선읽기.
   - macOS에서는 Objective-C 런타임(`objc_msgSend`)으로 `newBufferWithBytesNoCopy`를 호출해 무복사 Metal 버퍼를 만든다.
   - Python은 DLPack(장치 8 = Metal)으로 그 버퍼를 uint8 MPS 텐서로 받고, 가중치마다 오프셋 보기를 만든다(E015·E016과 같은 방법).
   - 이것은 버퍼 생성일 뿐 전용 커널이 아니므로 RS6(보류)와 무관하다.
5. **β와의 관계**: F 등급 텐서는 큰 저장공간의 보기다. β는 이를 공유 저장공간으로 보고 동면하지 않는다(0041 규칙). 이미 OS가 쓰기 없이 회수하는 등급이므로 동면할 이유도 없다.
6. **선읽기**: E016에서 효과가 엇갈렸다(여유가 있을 때는 없음, 압박 중에는 속도와 반응성을 맞바꿈). 그래서 기본으로 켜지 않고, E017의 쾌적 제어기가 다룬다.

## 구현과 검증

| 파일 | 내용 |
|---|---|
| `crates/memopro/src/residency.rs` | `FileMap`(mmap·mincore·pread 선읽기), macOS `metal_buffer()`(objc 런타임). Unix 전용 모듈(Windows는 빌드 제외) |
| `crates/memopro-py/src/lib.rs` | `_core.FileMap` 바인딩 |
| `python/memopro/residency.py` | DLPack으로 매핑을 uint8 MPS 텐서로, 골격(`init_empty_weights`, 생성 설정), 키 해석(접두사 보정·모델에 없는 키 건너뜀), 원본 매핑, 정렬 검사, F 캐시(작성 0600·목록·유효성·재사용), `load_file_backed` |
| `config.py`, `access/_load.py` | `residency` 설정과 `load(..., residency=)`. 명시 선택이므로 실패는 `ModeUnavailable` |

- **구현 중 확인한 것**
  - GPT-2 원본 safetensors는 텐서 위치가 4바이트 정렬이 아니라서 fp32 보기를 만들 수 없다.
  - 그래서 정렬이 안 된 원본은 `stored`도 **정렬된 F 캐시**(원래 형식 그대로의 사본)로 쓴다. 디스크 동의가 필요하다.
  - Qwen2.5 파일은 정렬돼 있어 원본을 그대로 매핑한다.
- **실제 모델** (M1, MPS)

  | 모델 | 경로 | 결과 |
  |---|---|---|
  | GPT-2 | 정렬 캐시 | 동의 없이는 거부. 동의하면 476MiB 캐시를 2.1초에 만들고, 재사용은 0.07초 |
  | Qwen2.5-1.5B bf16 | 원본 2.88GiB 매핑 | 0.73초 |
  | Qwen2.5-1.5B int4 | 캐시 1.12GiB | 만들기 9.4초, 재사용 0.13초 |

  모두 출력이 정상이었다.
- **테스트**
  - Rust 28개(`FileMap` 3개 추가; Metal은 장치가 없으면 Unsupported를 허용).
  - `tests/test_residency.py` 7개
    - 바인딩, 설정 검증, MPS가 아닐 때 거부, 정렬 검사.
    - MPS에서: 원본 매핑이 메모리 로드와 로짓 비트 동일, 가중치는 파일 전체 저장공간의 일부.
    - int4 캐시: 동의 없이 거부 → 만들기 → 재사용(동의 불필요), 로짓 비트 동일, 파일 권한 0600.
    - 원본이 바뀌면 캐시 무효.
  - 전체 Python 231개 통과.

## 논문 매핑

- **논문 C System**: RCR F 등급의 구현(Rust `FileMap` + Metal 무복사 + DLPack 보기), F 캐시의 유효성 규칙, 정책(디스크 동의·예산)과의 결합.
