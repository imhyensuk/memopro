# 0049. Colab 3차 검사 준비와 torch.compile 결함 수정

- **날짜**: 2026-09-25
- **유형**: milestone+experiment (검사 사전 등록, 결함 발견·수정)
- **상태**: 확정 (Colab 3차 실행 결과는 다음 기록)
- **관련 기록**: 0044 (Colab 노트북), 0045·0046 (Colab 1·2차), 0048 (가드)

## 배경

> "병합을 진행하고 CUDA 환경에서 테스트할 수 있는 구글 코랩 셀을 제작해."

- PR #2(alpha-prep)를 main에 병합했다(병합 커밋 `9052c47`).
- 0048의 가드는 기기와 무관한 코드지만 CPU·MPS에서만 확인했다. 그래서 같은 범주를 CUDA에서 확인하는 셀을 만들었다.
- 셀을 GPU에 보내기 전에 로컬 CPU에서 먼저 돌렸다. 이때 탐색용으로 넣은 `torch.compile` 검사에서 **새 결함**이 나왔다.

## 1. Colab용 wheel 빌드

- 이전처럼 Release 워크플로 수동 실행으로 wheel을 만들려 했다. 그런데 이 세션의 권한 검사가 이를 배포 작업으로 분류해 거부했다.
- 수동 실행은 배포하지 않지만(0044), 판정을 존중했다. 대신 로컬에서 교차 빌드했다.

```bash
pip install --target <작업공간>/zigpkg ziglang   # zig 0.16.0 (링커)
PYTHONPATH=<작업공간>/zigpkg ZIG_COMMAND="$PWD/.venv/bin/python -m ziglang" \
  .venv/bin/maturin build --release --target x86_64-unknown-linux-gnu --zig \
  --compatibility manylinux2014 -m crates/memopro-py/Cargo.toml -o <작업공간>/colab3
```

| 항목 | 값 |
|---|---|
| 기준 커밋 | `7a773eb` (브랜치 `colab-guards`) |
| 파일 | `memopro-0.1.0a1-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`, 744,306바이트 |
| SHA-256 | `c484ad3194d2a1a0e2780fffe8aea2a04eba9e9902bc663119b32e86386e4aee` |
| 도구 | rustc 1.98.1, maturin 1.15.0, zig 0.16.0 |

- CI의 wheel(maturin-action, manylinux 도커)과 빌드 경로가 다르다. 그래서 3차 실행은 이 빌드 방식의 확인도 겸한다.
- 노트북 첫 셀은 wheel 파일 이름과 `guards_0049`(0049 수정 포함 여부)를 기록한다. 버전 번호가 2차와 같으므로 `--force-reinstall`로 설치한다.

## 2. 3차 검사 사전 등록 (`examples/colab_cuda_check.ipynb`)

기존 구역(doctor, census_cuda, hibernate_gpt2, backward_in_flight)도 새 코드로 다시 돈다. 새 구역 3개는 **모든 검사가 `"ok": true`여야 통과**다. 기본 방법은 `host`이고, GPT-2 구역만 `auto`다.

| 구역 | 검사 | 통과 조건 |
|---|---|---|
| `guards_cuda` (14) | state_dict, torch_save(pickle 안에 memopro 흔적 없음), deepcopy, pickle_tensor, load_state_dict, move(`.cpu()`, `.half()`), count, step_foreach, step_fused, training(6스텝, 모델+AdamW 동면), interrupt(3번째 텐서에서 Ctrl-C), threads(8개 호출·6개 동시 `now()`), lifetime, clean_state | 값이 비트 단위로 같고, 장치·dtype이 맞고, 흔적이 남지 않는다. 스텝은 깨어 있는 쌍둥이와 비트 대조한다. training은 통제 실행(동면 없이 2회)으로 결정성을 함께 기록한다 |
| `guards_gpt2` (4) | 실제 GPT-2에서 동면 중 `generate`, `state_dict`, `save_pretrained` 후 다시 불러오기, backward와 step 사이 동면(파라미터는 `source`, 그래디언트는 `host`) | 토큰·logits·파라미터가 비트 단위로 같다 |
| `guards_compile` (7) | 원본을 동면시키고 컴파일된 래퍼 호출(inductor), `fullgraph=True`, `m.compile()`, `mode="reduce-overhead"`(CUDA graphs), 동면 뒤에 컴파일, 컴파일된 학습(모델+옵티마이저 동면), `torch.compile(opt.step)` | 컴파일 출력과 비트 단위로 같고, 그래프 분할 0, 새 그래프 0, 경고 0. 동면 뒤 컴파일은 값이 맞고 경고가 1회 |

- 모든 검사는 동면이 실제로 일어났는지(`h.nbytes > 0`)부터 확인한다. 아무것도 잠들지 않아 공허하게 통과하는 경우를 막기 위해서다.
- **로컬 사전 실행**(CPU, `compress`, inductor CPU): 25/25 통과. 이 실행은 셀 논리를 확인한 것이다. CUDA 고유 경로(host, CUDA graphs, triton)는 Colab에서 확인한다.

## 3. 발견한 결함: torch.compile

main(`9052c47`)의 가드 기준 증상이다.

| # | 상황 | 증상 | 심각도 |
|---|---|---|---|
| C1 | 컴파일된 모델(`torch.compile(m)`, `m.compile()`)을 동면 중에 호출 | `InternalTorchDynamoError: 'function' object has no attribute '__func__'`. dynamo가 `module.named_parameters.__func__`를 읽는데, 가드의 인스턴스 덮어쓰기가 일반 함수였다. 컴파일된 학습과 `torch.compile(opt.step)`도 같은 오류 | 🟠 크게 실패 |
| C2 | C1만 고쳤을 때(bound method + 추적 제외) | 값은 맞다. 그러나 깨우는 hook이 그래프 분할이 되고, dynamo가 **hook·파라미터 조건이 없는 eager 대체 캐시 항목**을 만든다(`skip_nnmodule_hook_guards=True`, 분석 재시작). 가장 새 항목부터 확인하므로 **이후 깨어 있는 호출도 전부 eager로 돈다** | 🔴 조용함(느려지고 끝자리가 바뀜) |
| C3 | `fullgraph=True` | 그래프 분할이 허용되지 않아 `Unsupported` 오류 | 🟠 |

C2의 증거(모델: `Linear(256)+GELU` 6층과 LayerNorm, CPU inductor):
- eager와 컴파일 출력의 최대 차이는 4.77e-7이다.
- 한 번 동면시킨 뒤로는 **모든 호출이 eager 출력과 같고 컴파일 출력과는 다르다.** 새 그래프는 0개였다.
- 컴파일된 학습(4스텝)에서 동면한 경우와 하지 않은 경우의 차이는 2.33e-10이었다. 이 값은 eager와 컴파일 학습의 차이와 **정확히 같다.**
- 캐시 항목을 출력해 보면 대체 항목은 입력, 전역 상태, 모듈 **타입**만 검사한다. `inner`(`wrap_inline`) 코드 객체는 모든 `torch.compile` 래퍼가 공유한다. 그래서 같은 클래스·같은 입력 모양의 **다른 컴파일 모델**도 이 항목을 탈 수 있다.

## 4. 시도했다가 버린 방법

| 시도 | 결과 |
|---|---|
| 추적되는 hook에서 잠든 텐서 크기를 먼저 읽어 조건으로 만들기 | dynamo가 인라인된 hook에서 분할을 만나면 분석을 다시 시작하고, hook에서 얻은 조건을 버린다. 대체 항목은 그대로였다 |
| `SkipFrame` | `apply_to_code=True`라서 공유 코드 객체의 **이후 모든 호출**을 건너뛴다. 모든 컴파일 모델이 eager가 된다 |
| 동면 중에는 `torch.compiler.set_stance("eager_on_recompile")` | 프로세스 전체 설정이다. 이 상태의 콜백은 "실행만"이라 다른 코드의 첫 컴파일도 막는다. 전역 변경 최소화(P4)에 어긋나 기각 |

## 5. 채택한 설계: 컴파일된 프레임에 들어가기 전에 깨운다

1. **호출하는 객체(래퍼)를 동면시키면** 래퍼 자신의 forward pre-hook이 `_call_impl`에서 dynamo(`forward`)보다 먼저 돈다. 수정 전에도 동작했다.
2. **원본을 동면시키면** 프로세스의 `OptimizedModule` 래퍼를 찾아 hook을 건다. `torch._dynamo.eval_frame`이 로드됐을 때만 찾는다(`import torch`만으로는 로드되지 않음을 확인). 비용은 GPT-2와 래퍼 13개, 객체 506,187개에서 93ms였다(`gc.get_objects()` 자체는 27ms).
3. **`m.compile()`**은 인스턴스의 `_compiled_call_impl`을 "깨우고 원래 것을 부르는" eager 함수로 바꿔 둔다. 깨울 때 원래 것으로 되돌린다.
4. **memopro가 볼 수 없는 컴파일 코드**(동면 뒤 컴파일, 모델을 부르는 컴파일 함수)는 값은 맞게 동작하고 **경고를 1회** 낸다. 호출 스택에 dynamo가 생성한 코드(`torch._dynamo.utils.orig_code_map`)가 있을 때만 경고한다. dynamo를 쓰지 않으면 이 확인 자체를 건너뛴다.
5. 깨우는 모든 함수는 `torch.compiler.disable`로 추적에서 뺀다. 이유 문구를 달아 dynamo 오류에 memopro 안내가 나오게 했다. 인스턴스 덮어쓰기는 bound method(`types.MethodType`)로 바꿨다.

## 6. 검증

| 항목 | 결과 |
|---|---|
| 탐색 11종(CPU inductor, `compress`) | 래퍼·원본·블록별 컴파일·`m.compile()`·`fullgraph`: 컴파일 출력과 비트 동일, 분할 0, 새 그래프 0, 경고 0. 동면 뒤 컴파일·컴파일 함수: 값이 맞고 경고 1회. 컴파일된 학습+동면이 동면 없는 학습과 비트 동일. 컴파일된 `opt.step` 정상 |
| 회귀 테스트 `tests/test_hibernate_compile.py` | 12개(backend `eager`, dynamo의 추적·캐시 결함이라 백엔드 무관). **수정본 12/12 통과, main 가드에서는 9개 실패** |
| 전체 | `pytest` **135 통과**(123 → 135), ruff 통과 |
| 0048 탐색 스크립트 재실행 | 결함 0/15, 0/17, 0/8 |
| 노트북 로컬 사전 실행 | 25/25 |

## 한계

- dynamo 내부에 기댄다: `OptimizedModule`, `_modules["_orig_mod"]`, `_compiled_call_impl`, `torch._dynamo.utils.orig_code_map`. 바뀌면 래퍼를 못 찾거나 경고가 사라진다. 그래도 값은 맞고 C2처럼 느려질 뿐이며, CI 테스트가 torch 갱신 때 이를 잡는다.
- 래퍼 탐색 비용은 프로세스의 객체 수에 비례한다(50만 개에 약 0.1초). dynamo를 쓴 프로세스의 모듈 동면에서만 든다.
- 동면 뒤에 컴파일하거나, 모델을 부르는 컴파일 함수는 첫 동면 호출 이후 그 프레임이 eager로 남는다(경고함). 권고: 호출 전에 `h.wake()` 하거나 컴파일된 래퍼를 동면시킨다.
- CUDA의 inductor·triton·CUDA graphs는 Colab 3차에서 확인한다. compiled autograd, DDP·FSDP와 compile의 조합은 시험하지 않았다.

## 논문 매핑

- **논문 B System**: "컴파일된 프레임에 들어가기 전에 깨운다"는 설계 규칙과 그 근거(분할 뒤 조건 없는 대체 캐시).
- **논문 B Threats to Validity**: 컴파일러 내부 API 의존과 볼 수 없는 호출자의 한계.
- **연구 방법론**: 검증 노트북을 GPU로 보내기 전에 로컬에서 먼저 돌려 결함을 찾았다. 실패한 시도 3가지와 기각 이유를 남긴다.
