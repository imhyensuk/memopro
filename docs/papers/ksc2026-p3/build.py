"""심사용(저자 정보 없음)·출판용 PDF를 paper.html에서 만든다(Chrome headless).

    python3 docs/papers/ksc2026-p3/build.py
"""

import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
KO = ('<div class="authors">임현석<sup>◦</sup><br>한신대학교 공공인재빅데이터융합학<br>'
      'sunlim0926@hs.ac.kr</div>')
EN = '<div class="authors">Hyensuk Im<sup>◦</sup><br>Hanshin University</div>'  # 영문 표기 확인 필요

src = (HERE / "paper.html").read_text()
for name, ko, en in (("review", "", ""), ("camera", KO, EN)):
    html = src.replace("<!--AUTHORS_KO-->", ko).replace("<!--AUTHORS_EN-->", en)
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "p.html"
        page.write_text(html)
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={HERE / f'ksc2026-p3-{name}.pdf'}", page.as_uri()],
                       check=True, capture_output=True)
    print(name, "ok")
