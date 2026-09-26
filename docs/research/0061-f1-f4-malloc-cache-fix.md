# 0061. F-E011-1 대응: F1(조각 단위 원본 대조)과 F4(macOS 할당기 캐시 안내·`run` 재시작)

- **날짜**: 2026-09-27
- **유형**: decision + milestone (구현)
- **상태**: 확정 (효과는 E014에서 잰다, 0062·0063)
- **관련 기록**: 0060 (E011 결과, 결함 F-E011-1과 선택지 F1~F5), 0038 (엔진 RS1~RS5, 다이제스트), 0039 (`source`)

## 배경

> "PR #6 병합하고 F1·F4 수정 후 벤치마크 진행해"

0060에서 두 가지를 확인했다.
- macOS libmalloc은 해제된 큰 블록을 캐시에 붙잡아 둔다. 그래서 동면이든 `del`이든 해제한 CPU 메모리가 다른 앱으로 바로 돌아가지 않는다.
- memopro `source`는 원본 대조를 위해 텐서마다 텐서 크기의 버퍼를 잡는다. 그래서 동면 직후 footprint가 오히려 늘었다(CPU +34%, MPS +62%).

사용자는 선택지 가운데 F1과 F4를 골랐다.

## 결정과 구현

### F1: 조각 단위 원본 대조

- **이전 방식**
  - 텐서 전체를 CPU 바이트로 만든다. MPS 텐서는 이때 통째로 복사된다.
  - 원본 파일 영역을 같은 크기의 새 버퍼에 읽고, 두 다이제스트를 비교한다.
- **새 방식** (`hibernate/_methods.py::_verified_digest`)
  - 텐서를 `VERIFY_PIECE` = 8 × `DIGEST_CHUNK` = 32MiB 조각으로 나눈다.
  - 조각마다 원본 파일의 같은 위치를 재사용 버퍼 하나에 읽는다. 파일 dtype이 다르면 텐서 dtype으로 바꾼다.
  - 텐서 조각과 **바이트 단위로 직접 비교**한다(`torch.equal`, uint8 보기). 장치 텐서는 이 조각만 CPU로 복사한다.
  - 같으면 조각의 청크 해시를 모은다. 조각 길이가 `DIGEST_CHUNK`의 배수라서 청크 해시를 합친 값은 **전체 다이제스트와 똑같다**. 그래서 깨울 때 쓰는 검증(`wake_source`)은 바뀌지 않는다.
- **Rust 변경**: `spill::digest_parts`, `spill::digest_combine`을 공개하고 바인딩 `engine_digest_parts`, `engine_digest_combine`을 추가했다. `digest`는 이 둘로 다시 썼으므로 값이 같다(Rust 테스트 `digest_parts_of_pieces_combine_to_the_whole`).
- **추가 메모리**: 텐서 크기와 관계없이 최대 조각 세 개(파일 버퍼, dtype 변환본, 장치 텐서의 CPU 조각). 32~96MiB다.
- **RS1과의 관계**: RS1("방출·복원은 한 번의 Rust 호출로, 큰 Python 객체 없이")은 방출과 복원에 대한 규칙이다. 동면 전 대조는 Python이 32MiB 조각을 돌며 Rust 읽기를 부른다. 큰 객체는 없고 메모리 상한이 있으므로 RS4의 취지에 맞다.
- **남은 것**
  - 깨울 때(`wake_source`)는 여전히 텐서 크기의 CPU 버퍼에 읽는다. CPU 텐서는 그 버퍼가 그대로 텐서가 되므로 낭비가 없다.
  - MPS 텐서는 깨운 뒤 CPU 사본이 해제되고, 그것이 캐시에 남는다. F4로 캐시를 끄면 사라진다.

### F4: macOS 할당기 캐시 안내와 `memopro run` 재시작

- `memopro.env.macos_malloc_cache_on()`: macOS이고 이 프로세스가 `MallocLargeCache=0`으로 시작하지 않았으면 참이다.
- **`memopro doctor`**: 참이면 메모에 안내를 낸다. 해제한 CPU 메모리가 압박이 올 때까지 캐시에 남으니, `MallocLargeCache=0`으로 Python을 시작하라는 내용이다.
- **동면 보고서**: CPU나 MPS 텐서를 동면하고 캐시가 켜져 있으면, `report()`에 같은 안내를 **프로세스당 한 번** "suggested"로 남긴다. 적용 항목보다 앞에 둔다.
- **`memopro run`**
  - 명령줄에서 실행했고, macOS이고, 캐시가 켜져 있으면 `MallocLargeCache=0`을 넣고 **자기 자신을 한 번 다시 실행**한다(`os.execve`). stderr에 알린다.
  - `--keep-malloc-cache`로 끌 수 있다. `--dry-run`에서는 재시작하지 않는다.
  - Python에서 `cli.main([...])`로 부르면 재시작하지 않는다(테스트나 내장 사용이 프로세스를 잃지 않도록).
- **대가**: 캐시를 끄면 큰 할당이 매번 OS에서 새 페이지로 온다. 할당과 해제를 반복하는 CPU 작업은 느려질 수 있다. E014에서 CPU 학습 스텝 시간으로 잰다(C7).

## 검증

- **Rust**: 테스트 25개 통과(`digest_parts` 추가). clippy 깨끗함.
- **`tests/test_hibernate_verify.py`** (4개)
  - 조각 해시를 합친 값이 전체 다이제스트와 같고, 비트 단위로 복원된다.
  - 파일이 bf16이고 텐서가 fp32여도 된다.
  - 마지막 조각 한 원소가 달라도 `source`를 거부한다.
  - 읽기 크기가 조각 크기를 넘지 않는다.
- **`tests/test_macos_malloc.py`** (6개)
  - 캐시 감지, doctor 메모, 보고서 안내(프로세스당 한 번), `run` 재시작 조건.
  - Python에서 부를 때는 재시작하지 않는다.
  - 실제 macOS에서 `memopro run`의 스크립트가 `MallocLargeCache=0`을 본다.
- **전체**: Python 208개 통과.

## 논문 매핑

- **논문 B System**: 조각 단위 검증(청크 해시의 결합성)으로 상한 있는 메모리에서 비트 동일성을 확인하는 설계.
- **논문 B Discussion**: 텐서 수준 도구가 할당기 정책(libmalloc 캐시)에 가로막히는 사례와 그 대응(환경 변수, 재시작). 대가(할당 비용)는 E014의 C7로 보고한다.
