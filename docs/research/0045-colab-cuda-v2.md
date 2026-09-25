# 0045. 배포 전 관문 V2: Colab T4 실제 CUDA 검증과 수정 F1·F2

- **날짜**: 2026-09-25
- **유형**: experiment(검증)+milestone
- **상태**: 1차 실행 확정 / F1·F2 수정 완료 / F2 확인 완료(→ 0046)
- **관련 기록**: 0044 (Colab 노트북), 0039 (β), 0037 (census V9 목표), 0041 (D2)

## 실행

- **환경**: Google Colab, Tesla T4, torch 2.11.0+cu128. memopro 0.1.0a1 wheel은 Release 워크플로 수동 실행 36135835739(Linux, 배포 없음)에서 받았다.
- **실행자**: 사용자가 `examples/colab_cuda_check.ipynb`를 실행하고 JSON을 그대로 전달했다.
- **원시 데이터**: `docs/research/data/v2_colab/colab_t4_run1.json`

## 1차 결과

| 항목 | 결과 | 판정 |
|---|---|---|
| doctor CUDA 여유 메모리 | memopro 15,527,116,800 = torch `mem_get_info` 15,527,116,800. 예산은 그 90% | ✅ 정확 |
| census CUDA 분류율 | **82.9%** (분류되지 않은 양 약 18MB) | ⚠️ 목표 90%(V9) 미달 |
| `host` 동면 (GPT-2 498MB) | CUDA 예약 메모리 511.7MB 감소, forward로 자동 복원, **비트 단위 동일** | ✅ |
| `auto` 동면 | `source` 자동 선택(HF 캐시), CUDA 511.7MB 감소, 호스트 RAM 증가 없음, SSD 쓰기 0, 비트 단위 동일 | ✅ |
| 역전파 도중 동면 (D2) | 그래디언트 동일. autograd가 저장한 가중치 2개는 깨어 있는 채로 둠(D1 규칙) | ✅ |
| 보고값 정확성 | memopro가 보고한 CUDA 회수량 = 노트북이 직접 잰 값(511,705,088) | ✅ |

**결론**: 실제 NVIDIA GPU에서 β의 핵심 약속이 성립한다. 원본 재읽기는 GPU 메모리를 비우면서 RAM·SSD를 쓰지 않고, 원본과 똑같이 복원한다. **V2는 통과**이며 남은 것은 아래 두 가지이다.

## 발견과 수정

| # | 발견 | 원인 | 수정 |
|---|---|---|---|
| **F1** | `host` 동면이 호스트 RAM을 약 498MB 더 쓰는데 보고는 `rss: 0` | 회수량을 `max(0, 전 − 후)`로 음수를 잘라 보고 → 비용이 숨겨짐(V7 위반) | `Handle.reclaimed`를 **부호 있는 풀별 변화량**으로 바꿨다(+ 회수, − 증가). 노트북 출력은 "freed cuda …; added rss …"이고, `report()` 상세에 풀별 freed/added를 적는다 |
| **F2** | census CUDA 분류율 82.9% | (가설) PyTorch가 캐싱 할당자로 잡는 **cuBLAS/cuBLASLt 작업 공간**(기본 설정 수십 MB 규모)이 분류 대상에 없었다 | `cublas_workspace_bytes()`: 작업 공간을 비우기 전후의 `memory_allocated` 차이를 재서 "framework workspace (cuBLAS)"로 결과에 넣고 분류율에 포함한다. 비워도 다음 행렬곱에서 다시 만들어진다. 내부 API(`torch._C._cuda_clearCublasWorkspaces`)가 없으면 측정하지 않는다 |

## 검증

- 로컬: 테스트 2개 추가(부호 있는 보고와 출력 문구, 작업 공간 표시). `pytest` 105 통과.
- **F2는 CUDA에서만 확인할 수 있다.** 노트북에 가설을 독립적으로 확인하는 셀을 추가했다: 워밍업 후 작업 공간을 직접 비워 줄어드는 양(`independent_workspace`)을 memopro의 `framework_workspace`와 분류율과 비교한다. **2차 Colab 실행 전까지 F2는 가설이다.**

## 한계

- 한 종류 GPU(T4), 한 torch 버전(2.11), 작은 모델에서의 결과이다.
- `host` 모드는 GPU 메모리를 호스트 RAM으로 옮기는 것이라 시스템 전체 메모리는 줄지 않는다. 이 점을 보고에서 드러내도록 F1에서 고쳤다.

## 논문 매핑

- **논문 B Evaluation**: CUDA 표(원본 재읽기 vs 호스트 이동, 비트 동일, 회수량 보고의 정확성).
- **논문 B Threats**: 할당자 수준 분류의 한계(프레임워크 작업 공간), 부호 있는 회수량 보고의 필요성.
