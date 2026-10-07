"""E048 (docs/research/0237): ``memopro run ...`` in this process, i.e. the given script inside a
memopro.enable session. The script (worker_data.py) writes the case's result.

    worker_memopro_run.py run --budget C --no-torch [--report-json PATH] worker_data.py image.py 1024
"""

import runpy
import sys

sys.argv = ["memopro", *sys.argv[1:]]
runpy.run_module("memopro", run_name="__main__")
