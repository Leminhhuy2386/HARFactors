# Cloud and local job files

Job files for `scripts/cloud/` (`kaggle_submit.py`, `colab_supervise.py`, `local_queue.py`). Each lists jobs:

- `{"models": [...], "dataset": "utd_mhad", "seed": 1, "splits": [0, 1]}`: a training job (`scripts/train.py`);
- `{"module": "src.models.classical.elastic_baseline", "args": [...]}`: a project module, run as `python -m`;
- `{"script": "scripts/<file>.py", "args": [...]}`: a script, relative to the project root.

The files here are the **historical** jobs that produced the imported runs in Edge_SSM (2026-10-08 … 10-13). Their
772 training jobs still run as written. Two kinds of script job refer to Edge_SSM's old flat layout:
- `"script": "elastic_baseline.py"` (7 jobs): now `{"module": "src.models.classical.elastic_baseline"}`;
- `"script": "synthetic_test.py"` (15 jobs): Edge_SSM's synthetic tasks, which are not part of this project.

Do not edit these files; write new job files for new runs. A VM clones the WHAR code at the paper commits and builds
the dataset caches itself (no data is uploaded).

Shipped to a VM: `scripts/train.py` and `src/{utils,training,models}` (`scripts/cloud/common.py: pack_code`); a module job
must live there.
