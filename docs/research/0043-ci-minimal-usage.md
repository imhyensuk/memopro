# 0043. CI 사용량 최소화 정책

- **날짜**: 2026-09-25
- **유형**: decision
- **상태**: 확정
- **관련 기록**: 0042 (첫 CI 실행)

## 사용자 결정

> "CI 사용량을 최소화하는 방식으로 승인할게."

비공개 저장소의 GitHub Actions는 무료 계정 기준 월 2,000분까지 쓸 수 있고, **macOS 실행 시간은 10배로 계산**된다. 첫 실행(0042)에서는 macOS 작업이 3개(Rust 1개, Python 2개)였다.

## 정책

| 이벤트 | 실행하는 작업 |
|---|---|
| PR | Linux만: Rust(fmt·clippy·test), Python 3.11·3.14, cgroup 512MiB 컨테이너 |
| main push, 수동 실행(`workflow_dispatch`) | 위 작업 + **macOS 작업 1개**(Rust 테스트 + Python 3.14 테스트) |
| 문서만 바뀐 경우(`docs/**`, `*.md`) | CI 생략 |
| 같은 브랜치에 새 커밋이 올라온 경우 | 진행 중이던 이전 실행 취소(`concurrency`) |
| 모든 실행 | Rust 빌드 캐시(`Swatinem/rust-cache`), pip 캐시 |

- macOS Python 3.11은 뺐다. Python 3.11 호환은 Linux에서, macOS 고유 경로(MPS 판정, statvfs, iopolicy)는 3.14 한 번으로 확인한다.
- clippy·fmt는 플랫폼과 무관하므로 Linux에서만 돌린다.

## 예상 효과 (대략)

- 이전: 한 번 실행에 macOS 작업 3개 × 약 2분 × 10배 + Linux 약 7분 ≈ **약 70분**
- PR 실행: 약 7분(macOS 없음, 캐시 적중 시 더 짧음)
- main 병합 실행: 약 7분 + macOS 약 2~3분 × 10배 ≈ **약 30분**

## 대가

- PR 단계에서는 macOS 회귀를 잡지 못한다. main에 병합한 뒤에야 드러난다. macOS가 걸린 변경은 병합 전에 수동 실행(`workflow_dispatch`)으로 확인할 수 있다.
- 저장소를 공개로 바꾸면(알파 배포 시 검토) Actions가 무료가 되므로 이 정책을 완화할 수 있다.

## 논문 매핑

- 해당 없음(개발 운영 기록). 도구 논문의 Artifact 절에서 CI 구성의 근거로 쓸 수 있다.
