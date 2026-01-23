# Conda/Mamba Adaptation Guide

This guide provides instructions on how to adapt the project setup to use `conda` or `mamba` for environment and dependency management instead of `poetry` or `pip` with `venv`.

## Rationale

Using `conda` or `mamba` can help prevent complex dependency conflicts, especially when packages have non-Python (e.g., C, C++) dependencies. `mamba` is a fast, drop-in replacement for `conda`.

## Steps

### 1. Create an `environment.yml` file

Instead of a `requirements.txt` or `pyproject.toml`, `conda` uses an `environment.yml` file to define the environment. Based on the `pyproject.toml` from the project's `sprint0.md`, here is the equivalent `environment.yml`:

```yaml
name: techdoc-genie
channels:
  - pytorch # for faiss-cpu
  - conda-forge
  - nodefaults
dependencies:
  - python=3.10
  - pip
  # Core Dependencies
  - langchain
  - langchain-openai
  - langchain-community
  - faiss-cpu
  - sentence-transformers
  - pypdf
  - python-dotenv
  - pydantic
  - pydantic-settings
  - fastapi
  - uvicorn
  - tiktoken
  - openai
  # Development Dependencies
  - pytest
  - black
  - ruff
  - mypy
  - jupyter
  # Pip dependencies (for packages not readily available on conda channels)
  - pip:
    - poetry # To read the pyproject.toml if needed, or if other packages are only on PyPI
```

**Note on Channels:**

*   `conda-forge` is the most common and comprehensive channel for community-maintained packages.
*   `pytorch` is often required for packages like `faiss-cpu`.
*   `nodefaults` tells conda to not search the default channel, which can speed up environment creation and improve consistency.

### 2. Create and Activate the Conda Environment

You can create the environment using either `conda` or `mamba`. `mamba` is recommended for its speed.

**Using `mamba` (Recommended):**

```bash
# First, install mamba if you haven't already
conda install mamba -n base -c conda-forge

# Create the environment from the yml file
mamba env create -f environment.yml
```

**Using `conda`:**

```bash
# Create the environment from the yml file
conda env create -f environment.yml
```

After creation, activate the new environment:

```bash
conda activate techdoc-genie
```

### 3. Adapting the `Makefile`

The `Makefile` can be simplified. The `setup` and `install` targets can be modified or removed, as the environment creation is now a single step.

**Original `Makefile` snippet:**

```makefile
setup:
	python -m venv venv
	. venv/bin/activate && pip install poetry
	. venv/bin/activate && poetry install

install:
	poetry install
```

**Adapted `Makefile`:**

You can replace the old `setup` and `install` with a single command to create the environment.

```makefile
.PHONY: setup install test format lint ...

# Remove the old setup and install targets.
# The environment is now managed by conda/mamba.
# You can add a target for environment creation if you like.

create-env:
    mamba env create -f environment.yml

test:
    pytest tests/ -v

format:
    black src/ tests/
    ruff check src/ tests/ --fix

lint:
    ruff check src/ tests/
    mypy src/

# ... rest of the makefile
```

Now, the setup process is:
1.  Run `make create-env` (or run the mamba/conda command directly).
2.  Activate the environment with `conda activate techdoc-genie`.
3.  Use the other `make` commands (`test`, `lint`, etc.) as before.

By following these steps, you can use `conda` or `mamba` to manage your environment, which should help in avoiding dependency issues.
