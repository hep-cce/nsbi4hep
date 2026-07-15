# Scaling Neural Simulation-Based Inference at High Performance Computing Centers for LHC analysis


## Installation
We use `uv` to manage our Python environment and dependencies.

```bash
uv venv
uv sync #--with dev,docs
source .venv/bin/activate
```
## Training NSBI models

```bash
uv run nsbi --help
```


#### References
* A. Held, J. Sandesara, "Introduction to NSBI", [link](https://indico.cern.ch/event/1656822/contributions/6963531/attachments/3277049/5855409/20260519_SBI_intro.pdf)
* ATLAS Collaboration, "An implementation of NSBI in ATLAS", [arXiv:2412.01600](https://arxiv.org/abs/2412.01600)
