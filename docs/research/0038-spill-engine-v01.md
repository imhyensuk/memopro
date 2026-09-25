# 0038. A1b 저장 엔진 구현: 원본 재읽기, 다이제스트, 방출 파일, 압축, buffer protocol 바인딩

- **날짜**: 2026-09-25
- **유형**: design+milestone
- **상태**: 확정
- **관련 기록**: 0027 (RS1~RS5), 0032 (H1~H4, P5), 0036 (제작 순서), 0025 (E008 프로토타입)

## 설계 결정

| # | 결정 | 이유 |
|---|---|---|
| E-a | **다이제스트 = xxh3-128**(`xxhash-rust`). 4MiB 청크별 해시를 병렬로 구한 뒤 길이와 함께 다시 해시한다. 스레드 수와 무관하게 같은 값이 나온다 | 목적은 손상·파일 변경 검출이며 적대적 공격 방어가 아니다. 비암호 해시 중 가장 빠른 부류이다. BLAKE3도 후보였으나 암호학적 강도는 필요 없다 |
| E-b | **읽기는 목적지 버퍼로 바로**, 쓰기는 원본 버퍼에서 바로 한다(`pread`/`pwrite` 병렬, 청크별 다이제스트를 같은 단계에서 계산) | RS1·RS5. **중간 버퍼가 없어** 추가 메모리가 전송 크기와 무관하다(RS4). E008b의 367MiB 문제는 구조적으로 사라진다 |
| E-c | **쓰기 전용 상주 스레드 풀**(최대 4개)을 두고, 풀 스레드의 I/O 우선순위를 낮춘다. macOS는 `setiopolicy_np(DISK, THREAD, UTILITY)`(libc 크레이트에 없어 직접 선언, 상수는 SDK 헤더에서 확인), Linux는 `ioprio_set` best-effort 최저(7) | H4. THROTTLE이 아니라 UTILITY를 쓴 이유: 경합 시에만 늦어진다. idle 클래스는 쓰기가 무기한 밀릴 수 있다. Linux NVMe 기본 스케줄러에서의 효과는 미확인 |
| E-d | 방출 파일: `create_new` + **0600**, 이름 `{pid}-{ns}-{counter}.mpspill`, 원시 바이트만(메타데이터·다이제스트는 호출자가 보관, pickle 없음). 쓰기 실패 시 파일 삭제. 내구성용 fsync는 하지 않는다 | P5. 방출 파일은 프로세스 종료·충돌 시 버리는 임시 자료라 fsync 비용이 필요 없다 |
| E-e | 읽기 시 다이제스트가 다르거나 파일이 기록 영역보다 짧으면 **`Error::Integrity`**를 낸다(Python `IntegrityError`) | 조용한 손상 복원을 막는다 |
| E-f | 압축(`codec::pack`/`unpack_into`): 청크 4MiB, **스레드마다 zstd 압축·해제 문맥과 작업 버퍼를 프로세스 수명 동안 재사용**(`thread_local`), 전역 rayon 풀 사용 | RS3. E008 프로토타입은 호출마다 풀을 만들었다 |
| E-g | **E008 프로토타입 제거**: Rust `compress`·`compress_reuse`·`spill_to_file`·`decompress_into`, Python `codec_*`(`bytes` 입력) | RS2 위반(0027 결정). E008 재현은 커밋 9fd9bd3에서 한다. 동결된 E008 스크립트는 현재 빌드에서 실행되지 않는다 |
| E-h | **바인딩은 buffer protocol**(`PyBuffer<u8>`, abi3 3.11 이상에서 사용 가능)로 C 연속 버퍼를 빌리고 GIL을 풀어 실행한다. 텐서는 `tensor.reshape(-1).view(torch.uint8).numpy()`로 복사 없이 넘긴다(bf16 포함). 쓰기 대상은 읽기 전용이면 거부한다 | RS2 |
| E-i | `hwinfo::process_rss()` 추가(sysinfo 프로세스 조회) | census·β의 실측 회수량 |

## 공개 API

- **Rust**
  - `spill::{digest, Engine::{digest, read_source_into, write, read_into}, SourceRef, SpillFile, Digest, DIGEST_CHUNK}`
  - `codec::{pack, unpack_into, shuffle, unshuffle, CHUNK}`
  - `hwinfo::process_rss`
  - `Error::Integrity`
- **Python `_core`**
  - `engine_digest`, `engine_read_source_into(path, offset, dst, expected=None)`, `engine_write(data, directory)`
  - `codec_pack(data, itemsize, level=1) → Compressed{raw_bytes, stored_bytes, unpack_into(dst)}`
  - `hwinfo_process_rss`

## 검증

| 검증 | 결과 |
|---|---|
| Rust 단위 테스트 | 21 통과. 다이제스트는 스레드 수와 무관하고 1비트 변화에도 달라짐. 방출 왕복이 비트 단위로 같고 권한 0600. 손상·잘린 파일은 `Integrity`. 원본 영역 읽기(오프셋·길이·크기 불일치 검사). 압축 왕복은 NaN·Inf·비정규 수와 2.5청크 경계를 포함해 비트 단위로 같음 |
| Python 연기 테스트 | 4MB 텐서: 다이제스트 16바이트, 압축 4.0MB → 3.37MB, 왕복 동일, 방출 파일 0600, 다이제스트 불일치 시 `RuntimeError(integrity)` |
| clippy·fmt | 경고 0 |

## 한계

- 방출 쓰기는 Unix 전용이다(Windows는 `NotImplemented`, 읽기는 구현됨).
- I/O 우선순위 조정의 효과는 측정하지 않았다(E009에서 함께 측정한다).
- 원본 재읽기 확인 단계에서 텐서 하나 크기의 버퍼를 최대 3개까지 잠시 쓴다(현재값 CPU 사본, 파일 버퍼, dtype 변환본). 메모리 상한은 "가장 큰 텐서 × 3"이며, 모델 전체 크기와는 무관하다.

## 논문 매핑

- **논문 B System**: 중간 버퍼 없는 전송 구조(RS1·RS4), 무결성 다이제스트, 저우선순위 쓰기 풀, 방출 파일 보안(P5).
