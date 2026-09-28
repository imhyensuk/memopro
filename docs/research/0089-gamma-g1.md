# 0089. G1 구현: γ는 실험 기능, macOS의 `memopro run`에서는 기본으로 끔

- **날짜**: 2026-09-29
- **유형**: implementation (사용자 결정)
- **상태**: 확정
- **관련 기록**: 0088 (E013a 결과: 쓸 만한 신호 없음, 규칙 3 → G1·G2), 0087, 0052 E6·E7

## 사용자 결정

> "PR #20 병합하고 G1 적용해"

- G1을 채택한다. γ를 실험 기능으로 표시하고, macOS에서 `memopro run`의 γ 기본값을 끈다.
- G2(경고에서는 손실 없는 조치만)와 E013b는 E010 데이터가 모인 뒤 다시 본다.

## 구현

- **`memopro._run`**
  - `run(..., elastic=None)`: `None`이면 플랫폼 기본값 `default_elastic()`을 쓴다. macOS면 끄고 그 밖이면 켠다.
  - 기본값 때문에 꺼진 경우 `report()`에 `elastic skipped`로 이유(`ELASTIC_OFF_NOTE`)를 남긴다.
  - `--dry-run`은 "elastic off (macOS default, 0088; --elastic)"로 보여 준다.
  - 명시적으로 `elastic=True`/`False`를 주면 그대로 따른다.
- **CLI**: `memopro run`에 `--elastic`을 더했다. `--no-elastic`과 함께 쓸 수 없다(상호 배타). 아무것도 주지 않으면 플랫폼 기본값이다.
- **`memopro.elastic`**: 모듈 설명에 "Experimental (0088)"과 근거를 적었다. `enable()`을 직접 부르는 사용법은 바꾸지 않았다.
- **문서**: README(한·영)의 γ 설명과 한계, CHANGELOG.
- **바꾸지 않은 것**: 수준별 예산 계수(경고 ×0.5, 위험 ×0.25), Linux PSI 문턱값, 셀·스텝 경계의 조치. γ를 켰을 때의 동작은 이전과 같다.

## 시험

- `tests/test_elastic_run.py`에 5개를 더했다(모두 16개).
  - 플랫폼별 기본값(macOS 끔, Linux 켬).
  - macOS에서 기본 실행은 γ를 켜지 않고 이유를 보고한다.
  - `elastic=True`면 켜고 보고하지 않는다.
  - CLI의 `--elastic`·`--no-elastic`·기본값 전달과 상호 배타.
  - dry-run 문구.
- 전체 Python 251개 통과, ruff 통과.

## 한계

- Linux에서는 γ가 여전히 기본으로 켜진다. PSI 기준은 보정하지 않았다. E013a는 macOS만 쟀다.
- 노트북에서 `%load_ext memopro` 등으로 γ를 켜는 경로가 있으면, 그것은 사용자의 명시적 선택으로 보고 바꾸지 않았다.

## 논문 매핑

- **논문 B**: 측정(E013a)에 따라 기본값을 보수적으로 바꾼 사례. "측정되지 않은 적응은 기본으로 켜지 않는다".
