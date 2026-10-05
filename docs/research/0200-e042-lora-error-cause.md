# 0200. E042 1차의 LoRA 오류 원인: Colab의 torchao 0.10과 peft 0.21 — 고치고 이어서 다시 재기

- **날짜**: 2026-10-06
- **유형**: 진단 + implementation
- **상태**: 확정. 사용자가 같은 노트북을 다시 실행해 이어서 잰다
- **관련 기록**: 0199 (E042 1차 결과), 0197 (사전 등록)

## 원인

- 사용자가 결과 폴더(`20261005-134520.zip`)를 보내 주었다. 경우 기록과 로그를 `docs/research/data/e042/run1/`에 옮겼다.
- **아홉 LoRA 경우(그냥 실행 포함) 모두 같은 오류**: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`.
  - Colab에 기본으로 깔린 torchao 0.10.0을, peft 0.21.0이 LoRA 층을 붙일 때(`dispatch_torchao` → `is_torchao_available`) 거부했다.
  - memopro와 상관없는 환경 충돌이다.
- 환경: Python 3.13.15, torch 2.11.0+cu130, transformers 5.17.0, peft 0.21.0, Tesla T4(sm 7.5).

## 데이터 작업 (0199의 측정 결함 확인)

- **Colab에서도 userfaultfd가 된다.** preload 점검(C)이 통과했다: 내보내기 436번, overruns 0.
- 네 작업 모두 큰 할당이 페이저로 갔다(2~638개). 그러나 한도가 잘못 커서(2.6GiB) 내보내기가 0번이었다. 0199의 결함(`ru_maxrss`) 그대로다.

## 고침 (`examples/colab_t4/`)

- `common.install_extras`: torchao가 0.16보다 오래되었으면 지운다. 이 노트북들은 torchao를 쓰지 않는다.
- `common.Run.done`: 이어서 할 때 `error` 상태인 경우도 다시 돌린다(지금까지는 `crashed`·`timeout`만). `oom`은 결과이므로 그대로 둔다.
- `cell_suite`: VmHWM 이전 측정으로 남은 데이터 경우(결과에 `maxrss_rusage_bytes`가 없는 것)는 지우고 다시 잰다.
- 그래서 같은 실행(`20261005-134520`)을 이어 가면 이미 통과한 생성·비전 추론은 건너뛰고, LoRA 아홉 경우와 데이터 여덟 경우만 다시 돈다.
  - 7B 흘려 쓰기 생성(약 1시간 반)을 다시 하지 않아도 된다.
  - 기준(0197)은 그대로다.

## 논문 매핑

- **Threats/재현성**: 클라우드 노트북의 사전 설치 패키지 충돌(torchao·peft)이 기준 실행까지 막은 사례. 환경을 기록하고 이어서 재는 구조로 대응했다.
