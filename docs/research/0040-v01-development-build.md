# 0040. v0.1 개발판 완성: 전 구간 제작 결과와 배포 전 검증 목록

- **날짜**: 2026-09-25
- **유형**: milestone
- **상태**: 확정 (개발판) / 배포 전 검증 대기
- **관련 기록**: 0036 (전 구간 제작 결정), 0034·0035 (뼈대, doctor), 0037 (census), 0038 (엔진), 0039 (β)

## 결과

0036의 "처음부터 끝까지"를 **v0.1 개발판**으로 완성했다. 설치해서 쓸 수 있는 상태이며, **배포는 하지 않았다.**

| 기능 | 상태 | 기록 |
|---|---|---|
| `memopro doctor` / `memopro.doctor()` | ✅ | 0035 |
| `memopro.census` (fast, light) + HF·Lightning 콜백 | ✅ | 0037 |
| `memopro.hibernate` (source·host·compress·bf16·spill, 핸들, 자동 복원, plan·suggest·status·enable) | ✅ (CUDA 실기 미검증) | 0039 |
| 노트북 통합 (`%load_ext memopro`, `%hibernate`, `%wake`, `%memopro status`, 유휴 제안, 자동 모드) | ✅ | 0039 |
| Rust 코어 (hwinfo, spill 엔진, codec, ledger) + buffer protocol 바인딩 | ✅ | 0035·0038 |
| 설정 (`configure`, `memopro.toml`, `MEMOPRO_*`), 오류 체계, fail-open, CLI | ✅ | 0034 |
| 5분 예제 노트북 (`examples/quickstart.ipynb`, 테스트에서 실행) | ✅ | 0039 |
| README(한·영), CHANGELOG, 알려진 한계 | ✅ | — |
| v0.2 (`load`·`optimize`·`train_session`·`check`), v0.3 (γ, `run`) | ⬜ `NotYetImplemented` | — |

## 품질 지표

| 항목 | 값 |
|---|---|
| 테스트 | Python 97 통과, Rust 21 통과(+1 컨테이너 전용 무시) |
| 린트·포맷 | clippy `-D warnings`, rustfmt, ruff check·format 통과 |
| 패키징 | wheel(abi3, Python ≥ 3.11)·sdist `twine check` 통과. 새 가상환경에 설치해 `memopro --version`, `memopro doctor --no-devices` 동작, `import memopro`가 torch를 불러오지 않음 확인. `cargo publish --dry-run` 통과(**업로드 안 함**) |
| import 비용 | 부작용 없음(테스트로 보장) |

## v0.1 완료 조건 (development-plan N1) 대비

| 조건 | 상태 |
|---|---|
| β ① 복원 결과가 비트 단위로 동일 | ✅ (source·compress·spill. bf16은 명시적 수치 변경) |
| β ② 실측 회수량 보고 | ✅ (RSS, MPS·CUDA 드라이버) |
| β ③ 동면 중 순간 메모리 증가를 텐서 단위로 제한 | ✅ 구조로 보장(텐서 하나씩 처리). 원본 확인은 최대 텐서 × 3(0038 한계) |
| β ④ 동면하지 않은 텐서의 속도 저하 없음 | ✅ 구조상 hook은 동면 중인 모듈에만 붙고 깨우면 제거된다. 별도 측정은 안 함 |
| β ⑤ 옵티마이저 묶음 규칙 | 🟡 **변경**: 묶음 대신 옵티마이저를 별도로 동면하고 `step()` 시 자동 복원한다(0036 B1의 정체성 유지 덕분에 참조가 깨지지 않음). 모델만 동면하면 옵티마이저 상태는 메모리에 남는다 |
| β ⑥ 시나리오 노트북에서 회수량·복원 지연 측정 | ✅ (0039 S1·S2, 예제 노트북) |
| β ⑦ 기준선 비교 | 🟡 `del`+`gc`+`empty_cache`와 다시 불러오기는 측정함. **OS 스왑·압축 대비(E011)와 pytorch_memlab CPU 이동 비교는 미실시** |
| census: 할당자 바이트 90% 이상 분류 | ✅ MPS에서 종료 시점 100%(0037). CUDA 미검증 |
| doctor: 컨테이너 한도 반영 | 🟡 CI 작업만 있음(Linux 미실행) |
| fail-open, 예제 노트북 CI 실행 | ✅ |
| 공개 시연 수치 (I3) | 🟡 M1만. Colab T4 미실시 |

## 배포 전 검증 목록 (0036: Gβ는 배포 전 검증)

| # | 항목 | 필요한 것 |
|---|---|---|
| V1 | **원격 저장소 연결 → CI 실행**: Linux·Python 3.11, cgroup 컨테이너 테스트 | GitHub 저장소 위치 결정(사용자) |
| V2 | **Colab T4 CUDA 검증**(I2): `host` 방식, census coverage, doctor CUDA | Colab 사용(사용자 계정) |
| V3 | **E010 β 실사용 수요**(Gβ): 사전 등록 → 수집 | 참여자 섭외, 필요 시 IRB 확인(사용자) |
| V4 | **E011 OS 스왑 대비**, **E009 압축 방출** | 사전 등록 후 실행(이 기기에서 가능. 다만 디스크 여유 8.6%라 방출 실험은 외장 디스크나 공간 확보 필요) |
| V5 | 버전 결정(0.1.0), 이름 선점과 배포 | 사용자 확인 |

## 한계

- 이 기기(M1 8GB)의 메모리 여유가 작고(1.3~2 GiB, 스왑 5~6GB 사용 중), 디스크 여유가 8.6%이다. 큰 모델 시연과 방출 실험은 이 기기에서 제한된다.
- 0.x 동안 API는 바뀔 수 있다(0031 I6의 안정성 약속은 배포 시점에 정한다).

## 논문 매핑

- **논문 B**: 이 기록의 표가 "구현 완성도와 검증 범위" 절의 뼈대가 된다. 남은 검증 V1~V4는 Evaluation과 Threats to Validity에 반영한다.
