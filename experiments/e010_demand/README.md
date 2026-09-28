# memopro E010 — 노트북 메모리 사용 조사 참여 안내 / Participant guide

## 한국어

평소 쓰는 Jupyter/IPython 노트북에서 **텐서와 모델이 메모리를 얼마나, 얼마 동안 쓰지 않은 채 차지하는지**를 기록합니다. memopro의 동면(β) 기능이 실제로 필요한지 판단하는 데 씁니다(연구 기록 0086).

**기록하는 것**
- 텐서를 담은 변수마다: 변수 이름의 **해시**(이름 자체는 저장하지 않음), 종류(Tensor, 모델 클래스 이름 등), 장치별 바이트, 마지막으로 쓴 뒤 지난 셀 수, 원본 가중치 파일이 이 컴퓨터에 있는 모델인지(예/아니요).
- 셀마다: 프로세스·GPU 메모리 사용량, OS의 가용 메모리와 스왑, 그 셀이 메모리 부족(OOM) 오류로 끝났는지(예/아니요).

**기록하지 않는 것**: 코드, 셀 내용, 값, 파일 경로, 모델 이름, 오류 메시지, 변수 이름.

**사용법**
1. `memopro_e010_probe.py` 파일 하나를 받습니다(memopro 설치는 필요 없습니다).
2. 노트북 첫 셀(또는 import 뒤)에서 `%run -i /경로/memopro_e010_probe.py`를 실행합니다.
3. 평소처럼 작업합니다. 셀마다 현재 폴더의 `memopro_e010_<세션>.jsonl`에 한 줄이 추가됩니다.
4. 지금 기록된 내용을 보려면 `%e010_summary`, 기록을 멈추려면 `%e010_stop`.
5. 작업이 끝나면(커널을 다시 시작하면 새 파일이 생깁니다) 파일을 **직접** 연구자에게 보냅니다. 도구는 아무 곳에도 전송하지 않습니다.

- 가능하면 **20셀 이상 작업한 세션**을 2~3개 보내 주세요(짧은 세션은 분석에서 빠집니다).
- 셀마다 수 밀리초~수십 밀리초가 더 걸립니다. 불편하면 언제든 `%e010_stop`.
- 보내기 전에 파일을 열어 내용을 확인할 수 있습니다(JSON 한 줄씩).

## English

The probe records, in your usual Jupyter/IPython notebooks, **how much memory tensors and models hold while they sit unused**. It informs whether memopro's hibernation (β) is needed (research log 0086).

**Recorded**: per variable holding tensors, a **hash** of its name (never the name), its type, bytes per device, cells since last use, and whether it is a model whose original weight files exist on this machine (yes/no); per cell, process and GPU memory in use, the OS's available memory and swap, and whether the cell ended with an out-of-memory error (yes/no).

**Not recorded**: code, cell text, values, file paths, model names, error messages, variable names.

**How to use**: get the single file `memopro_e010_probe.py` (no memopro install needed), run `%run -i /path/to/memopro_e010_probe.py` in the notebook, work as usual, and send the resulting `memopro_e010_<session>.jsonl` files yourself. `%e010_summary` shows what is recorded; `%e010_stop` stops. Sessions of 20 cells or more are analysed.
