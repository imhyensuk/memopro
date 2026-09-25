# 0035. A1a hwinfo와 N1a doctor 구현: 보수적 가용 메모리와 풀별 예산

- **날짜**: 2026-09-25
- **유형**: design+milestone
- **상태**: 확정 (macOS 검증 완료) / Linux 컨테이너 한도는 CI 작업만 작성, 미실행
- **관련 기록**: 0034 (뼈대), 0032 (순서, H4 여유 공간 기준), 0033 (K5), 0011 L1·L13 (풀별 예산, sysinfo), 0013 V12 (torch 없는 진단)

## 배경 / 동기

> "hwinfo와 doctor 구현을 진행해"

0032에서 정한 순서의 첫 구현 단계이다. doctor는 β 수요와 무관하게 쓸모가 있다. 이후 모든 기능(β, census, v0.2 구성 선택)이 쓰는 **풀별 예산 벡터**의 기반이기도 하다.

## 설계 결정

| # | 결정 | 근거 |
|---|---|---|
| D-a | **host 가용 메모리를 보수적으로 정의**한다: `total − used`. used는 OS가 다른 프로세스 메모리를 압축하거나 스왑하지 않고는 내줄 수 없는 양이다. macOS는 앱(익명) − 퍼저블 + wired + 압축기(Activity Monitor 기준), Linux는 `MemTotal − MemAvailable`이다. OS 자체 추정값은 `kernel_available_bytes`로만 보고하고 예산에는 쓰지 않는다 | 아래 "발견" 참조. sysinfo의 macOS `available_memory()`는 다른 앱의 active 익명 페이지까지 가용으로 센다 |
| D-b | 메모리·스왑·cgroup은 **sysinfo 0.39**(`default-features = false, features = ["system"]`)를 쓴다. cgroup은 sysinfo가 v2(`memory.max`·`memory.current`)와 v1(`memory.limit_in_bytes`)을 조상 경로까지 따라가며 읽는 것을 소스로 확인했다. 한도가 물리 메모리 이상이면 "한도 없음"으로 처리한다 | 0011 L13 "sysinfo를 쓰고 빠진 부분만 구현". 재구현 금지 |
| D-c | 디스크는 **`statvfs`**(libc)로 **실제 경로의 파일 시스템**을 잰다. `f_bavail × f_frsize`(일반 사용자 가용, 퍼저블 제외)를 쓴다. 방출 위치가 아직 없으면 가장 가까운 상위 폴더를 잰다. Windows는 `NotImplemented`(v0.1.x) | 마운트 목록의 접두어 대조보다 정확하다. `shutil.disk_usage`와 같은 정의이다 |
| D-d | **기본 방출 위치**: macOS `~/Library/Caches/memopro`, Linux `$XDG_CACHE_HOME/memopro`(기본 `~/.cache/memopro`), Windows `%LOCALAPPDATA%\memopro`. 폴더는 실제로 방출할 때만 만든다 | 사용자별 캐시 규약. doctor는 아무것도 만들지 않는다 |
| D-e | **예산 규칙** | |
| | host = 사용 가능 메모리(가용, 컨테이너 여유로 제한) × (1 − headroom) | U3 사전 예방 |
| | device(CUDA) = `mem_get_info`의 free × (1 − headroom) | |
| | device(MPS) = min(권장 한도 − 드라이버 할당량, host 사용 가능량) × (1 − headroom), `unified=True`(host와 같은 물리 메모리) | K5. 통합 메모리에서는 host가 부족하면 MPS 한도가 남아도 쓸 수 없다 |
| | disk = 여유 공간 − 전체 × `min_free_disk_fraction`(기본 20%). 쓰기 허용 여부는 예산이 아니라 `disk_writes` 정책이 정한다 | 0032 H4 |
| | 명시한 `budget` 설정은 device와 host를 함께 제한한다 | |
| D-f | **headroom 기본 10%**(설정 `headroom`, 0~0.9). 이 값은 **근거 있는 측정값이 아니라 초기값**이며 β·v0.2 실측으로 보정한다 | U3 "예산 여유분으로 사전 예방" |
| D-g | **장치 감지는 torch가 있을 때만**, doctor 호출 시에만 torch를 import한다. `devices=False`(`--no-devices`)는 torch를 건드리지 않는다. CUDA는 `mem_get_info`가 **CUDA 컨텍스트를 만든다**는 점을 코드에 명시했다. ROCm·XPU는 목록에만 올리고 예산은 만들지 않는다(fail-open) | 0013 V12, 0032 I4 |
| D-h | **공개 API**: `memopro.doctor(devices=True) → DoctorReport`(`summary()`, `to_json()`, `warnings()`). CLI는 `memopro doctor [--json] [--no-devices]`이며 종료 코드는 0이다. PyO3 바인딩은 `hwinfo_memory()`, `hwinfo_cpu()`, `hwinfo_disk(path)`이고 GIL을 풀고 실행한다 | 노트북 사용자도 함수로 쓸 수 있게 |
| D-i | **경고(Notes)**: 스왑 사용량, OS 추정값과 보수 값의 차이(전체의 10% 초과 시), 디스크 여유 공간이 기준 미만일 때 "이 디스크로 방출하지 않음", 예산 설정에 의한 제한, torch 관련 안내 | 사용자가 숫자의 의미를 오해하지 않게 한다(U6) |

## 발견: macOS "가용 메모리"의 두 정의

개발 기기에서 같은 순간에 잰 값이다(`vm_stat`, 페이지 16KB).

| 정의 | 값 | 구성 |
|---|---|---|
| sysinfo `available_memory()` = XNU memorystatus "available" | **3.27 GiB** | active + inactive + free. 커널의 `memory_pressure` "free percentage 44%"와 같은 기준 |
| **memopro `available_bytes`** (Activity Monitor 기준) | **1.55 GiB** | 8 GiB − (익명 − 퍼저블 + wired + 압축기 점유) |

같은 시점에 스왑이 **4.9GB** 사용 중이었다. 즉 시스템이 이미 메모리 압박 상태였다. 그런데도 앞의 정의는 3.27 GiB를 "가용"으로 보았다. 그 차이는 다른 앱이 실제로 쓰고 있는 active 익명 페이지이다.
- 이 메모리를 예산으로 잡으면, memopro가 할당할 때 OS가 다른 앱의 메모리를 압축하거나 스왑한다. 이것이 바로 memopro가 막으려는 상황이다(U3).
- Linux의 `MemAvailable`은 원래 익명 메모리를 제외한 추정값이다. 따라서 보수적 정의를 쓰면 두 OS의 의미가 일치한다.

이 차이는 **"8GB Mac에서 사용 가능한 메모리"를 어떻게 세느냐에 따라 예산이 두 배 넘게 달라진다**는 뜻이다. 논문 B의 설계 근거로 쓸 수 있다.

## 구현

| 위치 | 내용 |
|---|---|
| `crates/memopro/src/hwinfo.rs` | `memory()`, `cpu()`, `disk(path)`. `MemoryInfo { total, available(보수), kernel_available, swap_total, swap_free, cgroup: Option<CgroupMemory{limit, free}> }`, `usable_bytes()`, `swap_used_bytes()`. `DiskInfo::free_fraction()`. 패닉 없음 |
| `crates/memopro-py/src/lib.rs` | `hwinfo_memory`, `hwinfo_cpu`, `hwinfo_disk`. 오류 변환: NotImplemented → `NotImplementedError`, InvalidArgument → `ValueError`, Io → `OSError` |
| `python/memopro/env/` | `detect(devices=True, config=None) → Env`, `_torch.probe()`(CUDA·MPS·ROCm·XPU) |
| `python/memopro/orchestrator/budget.py` | `compute_budget(env, config) → Budget{device, host, disk, unified, device_name, headroom, disk_floor_bytes, capped_by_setting}` |
| `python/memopro/_doctor.py`, `cli.py` | `doctor()`, `DoctorReport`, `memopro doctor` |
| `python/memopro/config.py` | `headroom` 설정, `default_spill_dir()`, `spill_location()` |
| 의존성 | `sysinfo 0.39`(system 기능만), `libc 0.2` |

## 검증

| 검증 | 방법 | 결과 |
|---|---|---|
| 전체 메모리 | `hwinfo_memory().total_bytes` 대 `sysconf(PAGE_SIZE) × sysconf(PHYS_PAGES)` | ✅ 일치 |
| 가용 메모리 정의 | 같은 순간의 `vm_stat`으로 계산한 total − (익명 − 퍼저블 + wired + 압축기) 대비 | ✅ 전체의 5% 이내(실행 중인 시스템이라 두 스냅숏 사이에 값이 움직임). `available ≤ kernel_available` |
| 디스크 | `hwinfo_disk(path)` 대 `shutil.disk_usage(path)` | ✅ total 일치, free 차이 512MiB 미만. 없는 경로는 `OSError` |
| MPS 한도 | `doctor()`의 mps `limit_bytes` 대 `torch.mps.recommended_max_memory()` | ✅ 일치(5.33 GiB) |
| 예산 규칙 | 합성 환경으로 host 여유분, 컨테이너 제한, CUDA, 통합 메모리 두 경우, 디스크 기준 두 경우, 명시 예산, 미지원 장치 확인 | ✅ |
| import 비용 유지 | `doctor(devices=False)`가 torch를 불러오지 않음(별도 프로세스). 기존 `test_import_cost` 유지 | ✅ |
| CLI | `memopro doctor`, `--json`, `--no-devices` | ✅ 종료 코드 0, JSON 파싱 |
| 전체 | `cargo test`(20 통과, 1 무시: 컨테이너 전용), `pytest`(73 통과), clippy·ruff | ✅ |

**개발 기기 실측**(원시 데이터: `docs/research/data/n1a/doctor_m1.json`, 홈 경로 가림)
- 사용 가능 메모리: 1.5~2.1 GiB. 측정 시점마다 달라지며, OS 추정값은 2.7~3.3 GiB이다.
- 스왑: 4.9 GiB 사용 중.
- MPS 한도: 5.33 GiB. 장치 예산은 host에 묶여 약 1.4~2.1 GiB이다.
- 디스크: 여유 공간 9.2%로 20% 기준 미만이어서 **방출 예산 0**이다. doctor가 "이 디스크로 방출하지 않음"을 경고했다.

## 한계 및 향후 과제

- **Linux·cgroup은 로컬에서 검증하지 못했다.** 이 기기에 Docker가 없다.
  - 대신 CI에 작업을 추가했다: `rust:1` 이미지를 `--memory=512m`으로 실행하고, 무시 처리된 테스트 `cgroup_limit_matches_container`를 돌린다.
  - 원격 저장소를 연결하기 전까지는 실행되지 않는다.
  - GitHub 러너는 cgroup v2만 제공하므로 **v1은 sysinfo 소스 검토로만 확인**했다.
- **CUDA 경로는 합성 환경 테스트만** 있다. 실제 GPU 확인은 I2(Colab) 단계에서 한다.
- Windows 디스크 측정은 미구현(v0.1.x)이다.
- headroom 10%는 초기값이다(D-f).
- 사용 가능 메모리는 순간값이라 수십~수백 MB씩 움직인다. doctor는 스냅숏이며, 실행 중 변화는 γ(v0.3)의 영역이다.

## 논문 매핑

- **논문 B System 절**: 풀별 예산 규칙(D-e)과 통합 메모리 규칙(K5).
- **논문 B Motivation 또는 Discussion**: "가용 메모리"의 정의에 따라 8GB Mac의 예산이 두 배 넘게 달라진다는 발견(1.55 대 3.27 GiB, 스왑 4.9GB 사용 중).
- **Artifact**: doctor JSON은 실험 환경 기록(`env.json`)을 보완하는 표준 환경 스냅숏으로 쓸 수 있다.
