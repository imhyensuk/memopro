# 0042. GitHub 저장소 연결과 첫 CI 실행: Linux·컨테이너 검증 통과, macOS 가상머신 MPS 문제

- **날짜**: 2026-09-25
- **유형**: milestone+experiment (검증)
- **상태**: 확정
- **관련 기록**: 0040 (배포 전 관문 V1), 0035 (hwinfo·cgroup CI 작업), 0041

## 내용

| 항목 | 값 |
|---|---|
| 저장소 | `imhyensuk/memopro` (비공개). 사용자가 `gh` CLI로 로그인하고 `workflow` 권한을 추가한 뒤 생성 |
| 올린 브랜치 | `main`, `s2-library-skeleton`, `x1-first-experiments` |
| PR | [imhyensuk/memopro#1](https://github.com/imhyensuk/memopro/pull/1): v0.1 개발판 → main |

사용자는 저장소 이름으로 "memepro"를 요청했다. 패키지 이름(`memopro`)과의 일치를 권고하고 기본값을 알린 뒤 `memopro`로 만들었다. 이름은 나중에 바꿀 수 있다.

## 첫 CI 결과 (실행 36133529374)

| 작업 | 결과 |
|---|---|
| Rust (macos-14, ubuntu) | ✅ |
| Python 3.11·3.14 (ubuntu) | ✅ **Linux와 Python 3.11에서 처음 통과** |
| **Container memory limit (Linux, cgroup v2)** | ✅ **512MiB 컨테이너에서 `hwinfo`가 cgroup 한도를 정확히 인식**(0035 A1a 완료 조건 충족) |
| Python 3.11·3.14 (macos-14) | ❌ 2 실패 / 99 통과 |

**macOS 실패 원인**: GitHub macOS 러너(가상머신)에서는 `torch.backends.mps.is_available()`이 참이다. 그러나 할당을 하면 `MPS backend out of memory (MPS allocated: 0 bytes, ...)`로 실패한다. 테스트가 "사용 가능"이라는 응답만 믿고 MPS 경로를 실행했다.

**제품 관점의 의미**: `doctor`도 같은 가상머신에서 **쓸 수 없는 MPS를 장치로 보고하고 예산까지 잡았을 것**이다. 가상 macOS, 원격 데스크톱 등에서 사용자를 오도할 수 있는 결함이다.

## 수정

- `env._torch.mps_usable()`: MPS에 원소 4개짜리 텐서를 만들고 계산해 CPU로 가져오는 것까지 성공해야 참이다. 결과는 한 번만 계산해 둔다.
- `doctor`: 쓸 수 없는 MPS는 장치 목록에서 빼고 이유를 Notes에 적는다.
- census·hibernate의 메모리 측정은 MPS가 실제로 쓸 수 있을 때만 MPS 값을 읽는다.
- 테스트: MPS 테스트는 `mps_usable()` 기준으로 건너뛴다. "쓸 수 없는 MPS"를 가정한 테스트를 추가했다.
- 로컬: `pytest` 103 통과(이 Mac에서는 MPS가 쓸 수 있는 것으로 판정).

## 배포 전 관문 V1의 상태

- Linux 테스트, Python 3.11, 컨테이너 메모리 한도: **통과**.
- 두 번째 CI 실행 결과(macOS 포함 전체 통과 여부)는 이 기록의 부록이 아니라 PR #1에서 확인한다.

## 한계

- cgroup v1은 여전히 검증하지 않았다(GitHub 러너는 v2만 제공).
- 비공개 저장소의 GitHub Actions 사용 시간 한도(무료 계정 월 2,000분)가 있다. macOS 실행은 10배로 계산된다. CI 정책(macOS를 main에서만 돌릴지)은 사용자 결정 대기이다.

## 논문 매핑

- **논문 B Threats to Validity**: "장치가 사용 가능하다"는 프레임워크의 보고와 실제 사용 가능성의 차이. 가상 환경에서 memopro가 이를 어떻게 판정하는지를 쓴다.
- **Artifact**: 컨테이너 메모리 한도 인식이 CI에서 재현 가능하게 검증된다.
