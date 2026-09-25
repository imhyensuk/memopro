# 0044. 알파(0.1.0a1) 배포 준비: 버전, 배포 워크플로, Colab CUDA 확인 노트북

- **날짜**: 2026-09-25
- **유형**: milestone
- **상태**: 확정 (준비 완료) / 배포는 사용자 확인 대기
- **관련 기록**: 0040 (배포 전 관문 V1~V5), 0042 (첫 CI), 0043 (CI 정책), 0015 P2 (wheel 범위)

## 사용자 지시

> "진행해." (2단계 알파 준비의 2-1~2-3)

## 내용

| 항목 | 결정·구현 |
|---|---|
| 버전 | Cargo `0.1.0-alpha.1`, PyPI `0.1.0a1`(maturin이 PEP 440으로 바꿔 씀). `memopro.__version__`은 PEP 440 형식(`_pep440` 변환), `core_version()`은 Cargo 형식. 두 레지스트리 모두 **시험판**으로 취급되어 명시적으로 요청할 때만 설치된다(`pip install --pre`, cargo는 버전 명시) |
| 배포 워크플로 | ① **태그와 버전 일치 검사**(`v0.1.0a1` ↔ Cargo 버전) ② Linux x86_64·macOS arm64 wheel + sdist ③ **깨끗한 가상환경에서 wheel 설치 확인**(torch 없이 `--version`, `doctor --no-devices --json`, 버전 일치, torch 미로딩) ④ PyPI는 신뢰 게시(토큰 저장 없음, `pypi` 환경 승인 필요) ⑤ crates.io 첫 배포는 `CARGO_REGISTRY_TOKEN` 필요(신뢰 게시는 크레이트가 생긴 뒤에만 설정 가능), `crates-io` 환경 승인 필요 |
| 수동 실행 | **배포하지 않는다.** 기본값은 Linux wheel만 빌드한다(Colab용). macOS wheel은 선택하며, **job 단위 조건**이라 선택하지 않으면 macOS 러너가 아예 뜨지 않는다(비용 10배, 0043) |
| Colab 노트북 | `examples/colab_cuda_check.ipynb`: doctor CUDA 값과 `mem_get_info` 대조, census CUDA 분류율, GPT-2에서 `host`·`auto`(source) 동면 후 CUDA 예약 메모리 감소와 비트 단위 동일 복원, 역전파 도중 동면(D2)의 CUDA 확인. 구역마다 결과를 따로 기록하고 한 구역이 실패해도 계속하며, 마지막에 JSON을 출력한다. 이 기기(CPU)에서 끝까지 실행되고 CUDA 구역은 건너뜀을 확인했다 |
| 문서 | README(한·영) 상태를 알파로 표시하고 설치 방법(`--pre`)을 적음. CHANGELOG `0.1.0a1` |

## 배포 전에 남은 것 (사용자)

| # | 항목 |
|---|---|
| 1 | Colab 실행: 수동 실행으로 만든 Linux wheel을 올리고 노트북 실행, JSON 결과 전달 (V2) |
| 2 | pypi.org: 계정과 "pending trusted publisher" 등록(저장소 `imhyensuk/memopro`, 워크플로 `release.yml`, 환경 `pypi`) |
| 3 | crates.io: 계정, API 토큰을 저장소 비밀값 `CARGO_REGISTRY_TOKEN`으로 등록 |
| 4 | GitHub 저장소에 `pypi`, `crates-io` 환경 만들기(승인자 지정) |
| 5 | 저장소 공개 여부 |
| 6 | 태그 `v0.1.0a1` 푸시와 환경 승인 = 배포 실행 (**배포 직전 확인**) |

## 논문 매핑

- **Artifact**: 태그-버전 일치 검사, 깨끗한 환경 설치 확인, 토큰 없는 신뢰 게시는 재현 가능한 배포 절차의 근거이다.
