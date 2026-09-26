# 0051. γ와 `memopro run`의 중복성 재조사 (v0.3 착수 조건 0030 C5)

- **날짜**: 2026-09-26
- **유형**: survey
- **상태**: 확정
- **관련 기록**: 0029 D23·D24 (3차 중복성 검사), 0030 C5 (착수 조건), 0052 (제작 결정)

## 배경

> "아직 만들지 않은 것 제작을 전부 진행해."

v0.3(N3)의 착수 조건(0030 C5)은 "`memopro run`과 γ의 중복성 재조사를 먼저 기록한다"이다. 0029에서는 D24의 검색 범위가 좁았다. 이번에는 γ와 `run`에 더해, v0.2 접근 계층이 기대는 기존 도구(배치 크기 자동 탐색, 메모리 추정)도 함께 확인했다.

## 방법

웹 검색 9회(2026-09-26). 검색어:
- OS 메모리 압박(PSI·macOS memory pressure)과 PyTorch 학습 적응
- 메모리 탄력 런타임 논문(2024~2026)
- 코드 수정 없이 스크립트를 감싸는 메모리 도구, `from_pretrained` 패치
- OOM 뒤 배치 크기 자동 축소
- 통합 메모리에서 압박 신호에 따른 텐서 오프로드
- 스크립트를 감싸는 프로파일러(memray·scalene)

로컬에서는 기존 도구가 실제로 동작하는지 확인했다(torchao, bitsandbytes, torch MemTracker, huggingface_hub 메타데이터).

## 결과

### γ (OS 압박 탄력 런타임)

| 선행 사례 | 무엇을 하는가 | γ와의 차이 |
|---|---|---|
| Tri-Accel (2508.16905) | 학습 중 VRAM 사용량 되먹임으로 배치 크기 조절 | 자기 프로세스의 VRAM 측정. OS 수준 압박(다른 앱, 통합 메모리 스왑)을 보지 않는다 |
| eLLM (2506.15155) | LLM 서빙에서 유휴 활성값 청크를 빌려 큰 배치를 처리 | 서빙 엔진 내부의 메모리 관리. 범용 PyTorch 코드가 아니다 |
| LOCAL (2608.15241) | 에이전트 LLM 런타임이 메모리 압박을 KV 캐시 관리자에 신호로 보냄 | 스케줄러의 예상 여유분 기준, KV 캐시 한정 |
| accelerate `find_executable_batch_size`, Lightning `BatchSizeFinder` | OOM이 나면 배치를 줄여 다시 시도 / 사전 탐색 | OOM이 난 **뒤** 대응. OS 압박(스왑 폭주, 강제 종료)은 예방하지 못한다 |
| Linux PSI, macOS memory pressure | OS 신호 자체. 서버 작업 종료·이동 판단에 쓰인다 | 신호 제공자일 뿐 PyTorch 구성을 바꾸지 않는다 |
| SiliconBench (0029 D23) | "명시한 메모리 예산이 여유를 보장하지 않음"을 측정 | 문제 확인만. 해결책은 없다(γ의 동기) |

**결론**: OS 메모리 압박 신호로 **PyTorch 실행 구성을 되돌릴 수 있게 바꾸는 런타임**은 이번 조사에서도 찾지 못했다(not found in our survey). 가까운 것은 자기 VRAM 되먹임(Tri-Accel)과 OOM 뒤 재시도(accelerate)다. γ의 평가는 이 둘을 기준선으로 삼는다.

### `memopro run` (코드 수정 없는 실행)

| 선행 사례 | 무엇을 하는가 | 차이 |
|---|---|---|
| memray `run`, scalene `run` | 스크립트를 감싸 실행하며 **메모리를 측정** | 측정만 한다. 메모리를 줄이지 않는다 |
| transformers-patch (GitHub) | import 한 줄로 transformers를 패치해 메모리 절감 | import 추가 필요, 예산과 무관한 고정 패치 |
| Unsloth, Liger Kernel | 모델 코드 패치(커널 교체) | import·코드 변경 필요, 특정 모델군 |
| `PYTORCH_CUDA_ALLOC_CONF` 환경변수 | 할당자 설정을 코드 없이 바꿈 | 할당자 조정만 |
| accelerate `launch`, `torchrun` | 분산 실행기 | 메모리 적합이 목적이 아니다 |

**결론**: 스크립트를 수정 없이 감싸 **예산에 맞는 로딩 정책·압박 대응·진단**을 적용하는 명령은 찾지 못했다. 단, 구성 요소는 각각 선행 사례가 있다(측정 래퍼, import 패치). `run`은 접근 계층이므로 **신규성을 주장하지 않는다.** 차이는 "예산 기반 로딩 정책 + γ + census를 한 명령으로 묶는다"는 것이다.

### v0.2가 기대는 기존 도구 (재구현하지 않음)

| 용도 | 쓸 도구 | 로컬 확인(M1, torch 2.14) |
|---|---|---|
| 가중치 int8 | torchao `Int8WeightOnlyConfig`, bitsandbytes | torchao int8: CPU·MPS 동작. bitsandbytes 0.50.2: CPU·MPS에서 int8·int4 동작 |
| 가중치 int4 | bitsandbytes, torchao | torchao int4는 추가 의존성(`mslk`)이 없으면 불가 → 실제 시험으로 가용성을 판단해야 함 |
| 오프로드 | accelerate `device_map`·`max_memory` | (설치됨) |
| 메모리 추정 | torch `MemTracker` + `FakeTensorMode`(할당 없이 추정) | GPT-2 small 학습(배치 4, 길이 512) 최대치 추정 5,559MiB(파라미터 475, 활성값 4,299, 임시 785). 실제 할당 없음 |
| 모델 크기 | huggingface_hub `get_safetensors_metadata` | 다운로드 없이 GPT-2 파라미터 137,022,720개(버퍼 포함) |
| 기준선 | `accelerate estimate-memory`, `find_executable_batch_size` | 비교 실험에 사용 |

### 관찰

**이 Mac의 OS 압박 수준이 이미 "경고"다.** `sysctl kern.memorystatus_vm_pressure_level` = 2였고, 가용 1.44GiB, 스왑 4.96GB였다. GPT-2 small 학습을 MPS에서 시도하자 스왑이 폭주해 10분 안에 끝나지 않았고 중단했다. γ가 다루려는 상황 그대로다. 이후 로컬 검증은 작은 모델로 하고, 큰 규모는 Colab에서 한다.

## 논문 매핑

- **논문 B Related Work**: 메모리 압박 적응(Tri-Accel, eLLM, LOCAL, OOM 뒤 재시도)과 실행 래퍼(memray, transformers-patch). γ는 "OS 신호 기반 구성 변경"으로 위치시키고 신규성은 "not found in our survey"로만 적는다.
- **논문 B Motivation**: 8GB M1에서 이미 압박 경고 상태인 개발 환경(가용 1.44GiB, 스왑 4.96GB)의 실례.

## 참고문헌

- [@triaccel2025], [@siliconbench2026], [@ellm2025], [@local2026], [@memray], [@transformerspatch] — references.bib
