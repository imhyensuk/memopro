# 0034. 라이브러리 전체 뼈대 설계와 제작 (S2)

- **날짜**: 2026-09-25
- **유형**: design+milestone
- **상태**: 확정 — 설계 v0.3.9, 계획 v0.3.8에 반영
- **관련 기록**: 0016 (S1 걷는 뼈대), 0027 (RS1~RS5), 0032 (I4·I5·H1~H6·P5·I8), 0033 (연구 방향)

## 배경 / 동기

> "커밋을 진행하고 라이브러리 전체 뼈대 설계와 제작을 진행해."

S1(0016)은 빌드·배포 경로만 관통시킨 뼈대였다. 기능 구현(E010 → hwinfo → doctor → census)에 들어가기 전에 필요한 것이 두 가지이다.
- v0.1~v0.3의 **공개 API 전체를 실제 코드 구조로** 세운다.
- 기능과 무관한 **공통 기반을 먼저 구현하고 테스트로 고정**한다: 오류 체계, 설정, 방법 선택 정책, fail-open, 지연 import, CLI, 노트북 매직.

이렇게 하면 각 기능은 정해진 자리에 채워 넣기만 하면 된다.

## 설계 결정

| # | 결정 | 이유 |
|---|---|---|
| K-a | **만들지 않은 기능은 `NotYetImplemented`를 발생**시키고, 예정 버전과 설계 문서를 메시지에 담는다. 조용히 아무것도 하지 않는 빈 함수는 두지 않는다 | 사용자가 "동작한 줄" 오해하지 않게 한다(정직한 기대치, U6). `NotImplementedError`의 하위 클래스라 일반 처리 코드와도 맞는다 |
| K-b | **오류 계층**: `MemoproError` ← `InvalidArgument`(`ValueError`) ← `ConfigError`, `PolicyError`, `ModeUnavailable`(이유 + 대안), `NotYetImplemented` | 사용자가 memopro 자신의 오류만 골라 잡을 수 있다. H6 "조용히 바꾸지 않고 이유와 대안을 알린다"를 `ModeUnavailable`로 표현한다 |
| K-c | **지연 import**(PEP 562 `__getattr__`). `import memopro`는 Rust 확장과 최상위 모듈만 불러온다. torch·numpy·transformers·IPython은 쓰는 순간에 불러온다 | 0032 I4. 테스트로 보장한다(아래) |
| K-d | **설정 계층**: 기본값 < `memopro.toml`(또는 `$MEMOPRO_CONFIG`) < `MEMOPRO_*` 환경 변수 < `memopro.configure()`. import 시에는 읽지 않고, `get_config()`를 부를 때마다 다시 계산한다. 오류 메시지에 출처(파일·환경·configure)를 적는다. 거부된 `configure()`는 아무것도 바꾸지 않는다 | 노트북·서비스·CLI에서 같은 설정을 쓴다(0032 H6). 서비스는 코드 수정 없이 환경 변수로 제어한다 |
| K-e | **β 방법 선택 정책을 순수 함수로 먼저 구현**(`resolve_modes`). `auto`는 설정된 쓰기 없는 방법(`source → host → compress`) 뒤에 `spill`을 둔다. `bf16`은 자동 선택하지 않는다. `disk_writes="ask"`면 SSD 쓰기에 동의가 필요하고, `"never"`는 명시 요청까지 거부한다. 명시 모드는 대체 방법 없이 단독으로 시도한다 | 정책은 대상 객체와 무관하므로 지금 확정·테스트할 수 있다. `hibernate.now()`는 기능 구현 전에도 정책 위반을 즉시 알린다 |
| K-f | `hibernate_modes` 설정에는 쓰기 없는 방법만 넣을 수 있다. `bf16`(손실)과 `spill`(쓰기, `disk_writes`가 관장)은 거부한다 | 설정 파일 한 줄로 손실·쓰기 정책을 우회하지 못하게 한다 |
| K-g | **명시 핸들**(`Handle.wake()`)과 **매직**(`%hibernate`·`%wake`·`%memopro status`)을 둘 다 둔다. 매직은 memopro 오류를 출력만 하고 셀을 멈추지 않는다. `%hibernate NAME [--mode M] [--plan] [--spill]` | 0032 I5·H6, fail-open(U3) |
| K-h | **fail-open 실행기**(`orchestrator.fail_open`): 일반 예외는 보고서에 "failed"로 남기고 계속한다. `KeyboardInterrupt`·`SystemExit`는 삼키지 않는다 | U3. 이후 모든 기법 적용이 이 경로를 쓴다 |
| K-i | **CLI 종료 코드**: 0 성공, 1 memopro 오류, 2 사용법 오류, 3 미구현. `memopro run`은 옵션(`--budget`, `--disk-writes`, `--modes`)을 먼저 검증한다 | 스크립트·CI에서 구분할 수 있게 한다 |
| K-j | **Rust 코어 모듈 구조**: `error`(공통 `Error`·`Result`), `hwinfo`, `spill`(원본 재읽기·다이제스트 확인·방출 파일), `ledger`, `pressure`, `codec`(E008 프로토타입 유지). 미구현 함수는 `Error::NotImplemented { feature, planned }`를 반환하고 `unimplemented!()`는 쓰지 않는다 | 라이브러리가 패닉으로 사용자 프로세스를 죽이면 안 된다 |
| K-k | 순수 계산이라 지금 확정할 수 있는 것은 구현했다: `MemoryInfo::usable_bytes`(cgroup 한도 반영), `DiskInfo::free_fraction`(H4 20% 기준용), `EngineConfig::buffer_bound_bytes`(RS4 상한 공식), `Ledger`(방법별 바이트, SSD 누적 쓰기량 — 복원해도 줄지 않음, H5) | 이후 구현이 이 값을 테스트 기준으로 쓴다 |
| K-l | 다이제스트 알고리즘은 **정하지 않았다**(`Digest(Vec<u8>)`). A1b에서 속도·병렬성 비교 후 결정한다 | 근거 없이 정하지 않는다 |

## 구조

```
python/memopro/
├── __init__.py          공개 API (지연 import), load_ipython_extension
├── _errors.py           오류 계층 (K-b)
├── _units.py            크기 파싱·표시 ("20GB", "512MiB")
├── config.py            설정 계층 (K-d, K-f)                          ✅ 구현
├── report.py            보고서 (S1)                                   ✅ 구현
├── access.py            load · optimize · train_session · check       v0.2
├── cli.py, __main__.py  memopro doctor | check | run (K-i)            ✅ 골격 (명령은 미구현)
├── env/                 Env · HostMemory · Disk · Device, detect()    v0.1 A1a
├── orchestrator/        budget(Budget) · candidates(Configuration) · apply(fail_open ✅)
├── hibernate/           now · plan · wake · suggest · enable · status · Handle · PlanRow
│   └── _policy.py       resolve_modes (K-e)                           ✅ 구현
├── census/              record(mode = fast | light | deep) · Census   v0.1 / deep v0.2
├── elastic/             enable                                        v0.3
├── integrations/        ipython(✅ 매직) · hf · lightning (census 콜백, v0.1)
└── techniques/          base(S1) · native/ · integrations/ (어댑터 자리)

crates/memopro/src/
├── error.rs     Error { NotImplemented, InvalidArgument, Io }, Result      ✅
├── hwinfo.rs    MemoryInfo · DiskInfo, memory() · disk()                   v0.1 A1a
├── spill.rs     Engine · EngineConfig · SourceRef · SpillFile · Digest     v0.1 A1b (Gβ 후)
├── ledger.rs    Ledger · Entry · Location                                  ✅
├── pressure.rs  Level, current()                                           v0.3
└── codec.rs     E008 프로토타입 (A1b에서 정리)
```

- `pyproject.toml`: 콘솔 명령 `memopro`, 선택 설치 `hf`(torch·transformers·accelerate·safetensors), `notebook`(ipython). 버전 제약은 연동을 구현할 때 정한다.
- README(한국어·영어)와 크레이트 README를 0032 제품 방향(β + census, SSD 쓰기 기본 끔)에 맞게 고쳤다. 영어 README에 남아 있던 α(0.3)·γ(0.4) 표기도 바로잡았다.

## 실행한 명령과 결과

| 검증 | 명령 | 결과 |
|---|---|---|
| Rust 포맷·린트 | `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` | ✅ |
| Rust 테스트 | `cargo test -p memopro` | ✅ **17 통과** (기존 codec 6 + lib 1, 새로 error 2 · hwinfo 3 · spill 3 · ledger 1 · pressure 1) |
| Python 린트·포맷 | `ruff check`, `ruff format --check` | ✅ |
| 확장 빌드 | `maturin develop --release` | ✅ |
| Python 테스트 | `pytest -q` | ✅ **60 통과** (기존 4 + 새로 56) |
| import 비용 (I4) | 별도 프로세스에서 `import memopro` 후 `sys.modules`·환경 변수·작업 폴더 확인 | ✅ torch·numpy·transformers·lightning·IPython·tomllib 미로딩, 하위 모듈 8개 미로딩, 환경·파일 변화 없음 |
| import 시간 | `python -X importtime -c "import memopro"` | 누적 **약 16ms** (M1, Python 3.14, 참고값) |
| 콘솔 명령 | `memopro --version` / `memopro doctor` / `memopro run --modes source,bf16 app.py` | `memopro 0.0.1` / 종료 코드 3(미구현 안내) / 종료 코드 1(`bf16` 거부 이유 출력) |
| 실제 IPython | `start_ipython()` → `%load_ext memopro` → `%hibernate model_a --plan`, `%hibernate model_a`, `configure(disk_writes="never")` 후 `%hibernate model_a --spill`, `%wake`, `%memopro status` | ✅ 모두 안내 메시지만 출력하고 셀 유지. `never`일 때 SSD 방출 거부 메시지 확인 |

새 테스트 파일은 다음과 같다.
- `test_import_cost.py`
- `test_api.py`: 설계된 공개 이름 확인, 미구현 기능 15개의 예정 버전 확인
- `test_config.py`: 계층 우선순위, 잘못된 값 9종 거부
- `test_hibernate_policy.py`: H1·H2·H6 규칙
- `test_cli_and_magics.py`: 종료 코드, 매직 인자, 셀 유지, fail-open

## 한계 및 향후 과제

- 기능 자체(환경 감지, census, β, 방출 엔진)는 아직 없다. 공개 API의 **모양과 규칙**만 확정했다.
- API 이름과 인자는 기능을 구현하면서 바뀔 수 있다. 0.x 동안은 바뀔 때마다 기록하고, API 안정성 약속(0031 I6)은 v0.1 배포 시점에 정한다.
- 실제 IPython 검증은 이 기기(IPython 9.17)에서만 했다. Jupyter·Colab 커널에서의 확인은 I2 단계에서 한다.
- 다음 단계(0032 순서): E010 사전 등록(동료 참여 필요) → A1a `hwinfo` → N1a doctor.

## 논문 매핑

- **도구 논문(논문 B) System 절**: 오류 계층(미구현 명시, 이유와 대안 안내), 설정 계층, 쓰기 없는 방법 우선 정책, fail-open 실행기는 "신뢰를 위한 설계"의 구체적 근거이다.
- **Artifact**: 뼈대 단계부터 import 무비용·정책 규칙이 테스트로 고정되어 있다는 점은 재현성 평가 자료가 된다.
