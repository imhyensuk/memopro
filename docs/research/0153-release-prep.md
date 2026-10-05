# 0153. 첫 공개 배포 준비 (S4): 이름 확인, 메타데이터, 빌드·설치 시험 — 게시는 사용자 확인 대기

- **날짜**: 2026-10-05
- **유형**: implementation (배포 준비)
- **상태**: 확정. 게시는 사용자의 명시적 확인 대기
- **관련 기록**: 0148 (S4), 0097 (MIT OR Apache-2.0 오픈소스), 0044 (배포 작업 흐름), 0149 (한 줄 API)

## 사용자 지시

"CI 통과하면 병합하고 다음 단계 진행해." PR #41은 CI 통과 뒤 병합했다.

- 다음 단계 중 S1(7B)은 디스크가 20GiB라 진행할 수 없다(0151 보고와 같음).
- 그래서 S4(첫 공개 배포)를 **게시 직전까지** 진행한다. 게시(태그 push, PyPI·crates.io 업로드)는 되돌릴 수 없으므로 사용자 확인을 받는다(CLAUDE.md 규칙).

## 확인한 것

| 항목 | 결과 |
|---|---|
| 이름 | PyPI `memopro`: 배포본 없음(`pip index versions`). crates.io `memopro`: "does not exist"(API). 둘 다 비어 있다 |
| 판 | Cargo `0.1.0-alpha.1` = PEP 440 `0.1.0a1`(배포 작업 흐름이 태그와 대조) |
| 휠 | `maturin build --release`: `memopro-0.1.0a1-cp311-abi3-macosx_11_0_arm64.whl`(830KB, 파일 62개, 라이선스 2개 포함) |
| 휠 설치 시험 | torch 없는 새 venv(Python 3.12)에 설치. `memopro --version`, `memopro doctor --no-devices --json`, `import memopro`(torch·numpy를 불러오지 않음), `finetune`·`generate`가 이름 목록에 있음, `memopro.rt.Runtime(budget="64MiB")` 생성 |
| sdist | `maturin sdist`: 196KB(python 54, crates 22, 설정·라이선스). 새 venv에서 소스 빌드·설치 성공 |
| 크레이트 | `cargo publish --dry-run -p memopro`: 패키징·컴파일 성공(업로드는 모의라 중단) |

## 고친 것

- `pyproject.toml`
  - 설명을 0148 목표로 바꿨다(옛 "memory relief and redundancy diagnostics" → "Run work that exceeds your machine's memory losslessly, within a guaranteed memory ceiling …").
  - 키워드를 바꿨다.
  - 한 줄 API용 선택 의존성 `llm`(torch, numpy, transformers, accelerate, safetensors, peft, torchao)을 더했다.
- `crates/memopro/Cargo.toml`: 설명과 키워드를 런타임 중심으로 바꿨다.
- `crates/memopro/README.md`(crates.io에 표시)
  - 0.0.x 골격 설명을 지금 모듈로 바꿨다: `rt`, `rt::pager`, `hwinfo`, `pressure`, `residency`, `spill`, `codec`.
  - C ABI와 Python 바인딩의 위치를 적었다.

## 게시에 필요한 사용자 행동 (Claude Code가 대신할 수 없음)

1. **PyPI Trusted Publishing**: pypi.org에서 pending publisher를 등록한다. 저장소 `imhyensuk/memopro`, 작업 흐름 `release.yml`, 환경 `pypi`.
2. **crates.io 토큰**: 첫 배포는 저장소 비밀 `CARGO_REGISTRY_TOKEN`이 필요하다. crates.io의 trusted publishing은 크레이트가 생긴 뒤에만 설정할 수 있다.
3. **GitHub 환경** `pypi`·`crates-io`: 승인자를 지정한다. 작업 흐름이 거기서 멈추고 승인을 기다린다.
4. **저장소 공개 여부**: GitHub 저장소는 지금 private이다. PyPI의 sdist와 crates.io에는 소스가 공개된다. 그런데 문서의 저장소 링크(크레이트 README 등)는 private이면 열리지 않는다. 0097(오픈소스)에 따라 공개로 바꿀지 결정이 필요하다.
5. 위가 끝나고 확인을 받으면 Claude Code가 `v0.1.0a1` 태그를 push한다. 그러면 휠(Linux x86_64, macOS arm64), sdist, 크레이트가 만들어지고 환경 승인 뒤 게시된다.

## 논문 매핑

- **Artifact**: 공개 배포물(PyPI·crates.io)이 논문 B의 재현 가능 산출물이 된다.
