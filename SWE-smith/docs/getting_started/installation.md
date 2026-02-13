# Installation

For the latest stable release

```bash
pip install swesmith
```

For the latest development version

```bash
git clone https://github.com/SWE-bench/SWE-smith
cd SWE-smith
conda create -n swesmith python=3.10; conda activate swesmith
pip install -e .
```

If you plan to contribute to SWE-smith, please also perform:

```bash
pre-commit install
```