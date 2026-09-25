# memopro

> **노트북과 학습 중에 잠든 메모리를 되찾고, 낭비되는 메모리를 보여 준다.**
> PyTorch 개발자를 위한 메모리 회수·진단 라이브러리 (Rust + Python). 불러온 모델은 SSD에 쓰지 않고 원본에서 복원하며, 모든 절감량은 실측으로 보고한다.
> 이후 버전(v0.2~)에서는 내 기기의 실제 가용 메모리에 맞춰 모델을 불러오고 학습하는 편의 기능을 더한다.

- **Python** (PyPI `memopro`): PyTorch 개발자를 위한 한 줄 인터페이스
- **Rust** (crates.io `memopro`): 프레임워크와 무관한 메모리 코어 (환경·예산 감지, 원본 파일 재읽기와 해시 확인, 텐서 원장, 메모리 상한이 보장된 방출)

> 여기서 "메모리"는 에이전트·대화 기억(agent memory)이 아니라 GPU/RAM **하드웨어 메모리**를 뜻한다.

상태: **S2 라이브러리 전체 뼈대 완료** (2026-09-25) — 공개 API 구조와 공통 기반(설정, 방법 선택 정책, fail-open, CLI, 노트북 매직)은 동작한다. 아래 기능은 아직 구현되지 않았으며, 호출하면 `NotYetImplemented`가 예정 버전을 알려 준다.

---

## 1. 목표와 대상

| # | 목표 |
|---|---|
| 궁극 | 메모리가 부족한 **PyTorch 개발자** 누구나, 자기 프로젝트·서비스·개발·학습에서 큰 메모리를 요구하는 모델을 쉽게 쓴다 |
| G1 | AI 개발·학습·활용 **전 단계**에서 메모리 부담 최소화 |
| G2 | 극한의 효율로 **더 작은 메모리 환경**에서 구동 |
| G3 | 메모리 **하드웨어 병목** 극복 — **장기 과제**: v0.1~v0.3은 G1·G2 중심, G3 후보는 센서스 연구 결과에서 탐색 (0033) |

- **대상**: 직접 만든 모델, 이미지 생성·비전·오디오 모델, Python 코드 안의 LLM, 파인튜닝, 연구 코드, Python 기반 서비스를 다루는 PyTorch 개발자
- **대상 아님**: 코드 없이 LLM 앱을 쓰려는 최종 사용자

## 2. 이렇게 쓴다 (설계안)

**v0.1: 진단·회수**
```bash
pip install "memopro[torch]"
memopro doctor        # 내 환경의 풀별 예산: 장치·호스트 RAM·디스크 (컨테이너 한도, Apple Silicon 한도 반영)
```
```python
import memopro

%load_ext memopro     # 한동안 안 쓴 텐서·모델과 회수 가능량을 제안
%hibernate old_model          # β: SSD에 쓰지 않는 방법부터(원본 재읽기 → GPU→RAM → RAM 압축), 다시 쓰면 복원
%hibernate old_model --plan   # 방법별 회수량·복원 시간·SSD 쓰기를 미리 비교 (--mode 로 직접 선택)
%wake old_model               # 바로 복원. SSD 쓰기는 허락할 때만 (disk_writes="ask")

with memopro.census.record(model, optimizer) as c:    # 메모리가 어디에 쓰이고, 얼마나 중복(낭비)인가
    loss = model(**batch).loss; loss.backward()
print(c.summary())
```

**v0.2: 내 예산에 맞추기**
```python
model = memopro.optimize(model, goal="infer")          # 예산에 맞는 구성 자동 선택·적용
with memopro.train_session(model, optimizer, batch_size=32) as s:
    for batch in s.batches(loader):
        s.step(model(**batch).loss)
memopro.report()   # 무엇을 적용했고, 실측으로 얼마나 줄였고, 품질·속도는 어떻게 변했는지
```

**v0.3**: γ 탄력 런타임과 코드 수정 없는 실행 `memopro run app.py` (α 잔차 고정점 체크포인팅은 X1 실험에서 기각 — 0018)

## 3. 구조: 2계층

| 계층 | 내용 | 가치 |
|---|---|---|
| **범용 접근 계층** | 환경 감지 → **풀별 예산 벡터**(장치·호스트·디스크) → 후보 구성 중 예산에 맞는 것을 선택 → 적용 → 실측 보고. 검증된 기존 기법(양자화, 오프로드, 지연 로딩, 체크포인팅 등)은 **재구현하지 않고 연결** | 누구나, 바로 |
| **연구 코어** (memopro 고유) | **β** 유휴 텐서 동면 · **census** 메모리 중복도 측정 · **γ** OS 메모리 압박 탄력 런타임 (~~α~~ 0018 기각) | 새로움, 학술 기여 |

### 기존 도구와의 관계 (0029 중복성 검사)
memopro의 고유 영역은 연구 코어(β·census·γ)이다. 나머지는 기존 도구를 **재구현하지 않고 연결·통합**하며, 신규성을 주장하지 않는다.

| memopro 기능 | 비교 대상 기존 도구 | memopro의 차이 |
|---|---|---|
| `check` | vram-check, accelerate estimate-memory, llmfit | Apple Silicon·통합 메모리 한도와 컨테이너 한도, 장치·호스트·디스크 풀별 판정, 작은 배치 시험 실행으로 실측 외삽, 판정 결과를 바로 적용 |
| `train_session` | ProTrain, AutoCheckpoint, Unsloth, HF `auto_find_batch_size` | 소형·MPS 환경 포함, 기존 기법 연동과 충실도 3분류, 실패 시 대체와 실측 보고 |
| β 방출 엔진 | TensorNVMe, DeepNVMe, kvikio | macOS 지원, 구조적 메모리 상한, 노트북 동면과 결합 |
| census 정밀 모드 | AIMET QuantAnalyzer, HAWQ 계열 | 추론 층별 양자화 민감도가 아니라 학습 상태 범주별 비트 낭비 측정 |

### 연구 코어 요약
- **β 유휴 텐서 동면** (v0.1): 노트북·REPL에서 한동안 쓰지 않은 텐서와 모델을 **제안**하고, 사용자가 한 줄로 압축하거나 SSD로 내보내며, 다시 쓰는 순간 복원한다. 1순위 환경은 CUDA GPU 메모리와 스왑 없는 환경이다(0015 P3). 참조 관계를 분석해 안전한 대상만 고르고, 동면 과정에서 메모리가 튀지 않게 청크 단위로 처리하며, 절감량은 **실측**으로 보고한다.
- **census** (v0.1): 메모리가 "어디에" 쓰이는지를 넘어, 저장된 텐서가 실제로 담고 있는 정보량 대비 **얼마나 중복(낭비)인지**를 측정한다.
- ~~**α 잔차 고정점 체크포인팅**~~ — **X1 실험에서 기각**([0018](docs/research/0018-alpha-e001-e003-results.md)): 사전학습 GPT-2의 블록은 모든 층에서 비수축(ρ > 1)이고, 역순 체인은 위층 오차를 교정 없이 누적했다. 교정은 오히려 그래디언트를 나쁘게 만들었다(4비트: 교정 전 코사인 0.997 → 교정 후 0.786).
- **γ 탄력 런타임** (v0.3): 실행 중 메모리 압박에 따라 구성을 바꾼다. OOM 대신 단계적으로 열화한다.

### X1 첫 실험의 주요 결과 (0018~0020)
- α 가설 H1·H2·H3 모두 기각 → α 트랙 종료 (사전 등록 규칙).
- 학습 중 fp32 텐서의 무손실 중복은 약 14%에 불과하고, OS 방식(페이지 단위 LZ4) 압축은 fp32 텐서에 사실상 효과가 없다(1.00배) → β의 CPU 기본 모드를 **SSD 방출**로 변경, bf16 손실 모드는 명시적 선택(fp32 대비 2.75~3.8배).
- 노트북 유휴 메모리 계측 도구를 만들고 검증했다. 실제 수요 데이터는 사용자·동료 노트북에서 수집할 예정이다.
- 후속 실험(0024·0025): Muon 옵티마이저는 M1 소배치에서 3배 느려 후보에서 제외. 단순한 Rust 코덱은 Python 스레드보다 느렸고, 설계를 바꾼 Rust(버퍼 재사용·파이프라인·복사 제거)는 1.49배 빨랐다 → Rust 코어 설계 규칙 RS1~RS5 채택(0027), β 방출 기본값은 무압축.

## 개발 환경 (S1)

```bash
python3 -m venv .venv && .venv/bin/pip install "maturin>=1.9,<2" pytest ruff
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release   # Rust 확장 빌드 + 설치
.venv/bin/pytest -q                                           # Python 테스트
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test -p memopro
.venv/bin/maturin build --release --out dist                  # wheel (abi3, Python ≥ 3.11)
```

구성: Rust 코어 `crates/memopro`(crates.io) · PyO3 바인딩 `crates/memopro-py`(비공개) · Python 패키지 `python/memopro`(PyPI) · 실험 `experiments/`(환경 자동 기록 하네스 포함)

## 4. 설계 문서

- [architecture.md](docs/design/architecture.md): 2계층 구조, 원칙 U1~U9, 풀별 예산 벡터, 후보 구성 선택, 충실도 3분류, β·census 상세 설계
- [use-cases.md](docs/design/use-cases.md): 환경별 시나리오와 적용 범위
- [development-plan.md](docs/design/development-plan.md): 트랙 N(고유 기법) / A(범용 접근) / R(연구), 관문, 완료 조건, 위험

## 5. 로드맵

| 단계 | 내용 | 배포 | 상태 |
|---|---|---|---|
| S0 | 조사, 방향 설정, 설계, 검증 (연구 기록 0001~0015) | | ✅ |
| S1 | 기반 구축 + 걷는 뼈대: git, 가상환경, Cargo workspace, PyO3·maturin, CI 설정, `Technique` 뼈대, 실험 하네스, wheel·sdist·crate 패키징 확인 | 0.0.1 (로컬 빌드만, 미배포) | ✅ |
| S2 | **라이브러리 전체 뼈대** (0034): 공개 API, 오류·설정 계층, β 방법 선택 정책, fail-open, CLI, 노트북 매직, Rust 모듈 구조. 테스트 Python 60·Rust 17 | | ✅ |
| X1 | **첫 실험** (0015 P1): α E001~E003(→ 기각), 텐서 중복도·OS 압축 기준선 E005, 노트북 유휴 계측 도구 E006 | | ✅ (0018~0021) |
| E010 | β 실사용 수요 수집 (사전 등록, 동료 3~5명) → **◆ Gβ** (0032) | | |
| A1a·N1a·N1b | hwinfo → **doctor** → **census**(빠른 + 정밀 경량, HF·Lightning 콜백) | | |
| A1b·N1c | (Gβ 통과 시) 방출 엔진(원본 재읽기·해시 우선, RS1~RS5, SSD 정책) → E009·E011 → **β** | | 설계 규칙 확정, 프로토타입 codec 완료(E008) |
| N1 | **doctor + census + β** 통합, 5분 시연 노트북, 공개 시연 수치 | 🚀 v0.1.0 (crates.io + PyPI) | |
| R2 | 메모리 센서스 연구 (census와 코드 공유, 연구 주력 후보 — 0021 Q2) | | |
| A2 | 범용 접근: `optimize`, `train_session`, `load`, `check`, 생태계 통합 | 🚀 v0.2.0 | |
| ~~◆ Gα → N2~~ | ~~α 등록~~ — 0018 기각으로 취소 | | ❌ |
| ◆ Gγ → N3 | γ + pressure + `memopro run` | 🚀 v0.3.0 | |
| S6 | 안정화, 문서 사이트(영어·한국어) | 🚀 v1.0.0 | |

## 6. 원칙 (요약)

설정 없이 동작 · memopro 자신의 오류로는 멈추지 않고 OOM은 사전 예방 · 약속한 품질 이상으로 손실을 내지 않음 · 충실도 3분류(정확 / 수치 변경 / 의미 변경은 제안만) · 예상 메모리와 대략적 속도를 먼저 알려줌 · 절감량은 실측으로 보고 · 풀별(장치·호스트·디스크) 실제 가용 메모리 기준 · 기존 기법은 재구현하지 않고 연결

## 7. 연구 기록

모든 단계의 결정·실험·결과는 논문과 연구보고서 작성을 위해 [docs/research/](docs/research/)에 기록한다.
방향의 변화(플래너 → 고유 기법 → 범용 2계층 → 검증 → 대상 재정의 → 실험 → α 기각)도 0001~0021에 남아 있다.

## License

MIT OR Apache-2.0 (둘 중 선택). S1에서 권고안을 적용했으며 변경 가능하다 (0015).
