# Installation

OMpy targets Python 3.10+. Until it is published on PyPI you install it from a
local checkout. Two supported workflows are outlined below.

## Option 1 — Virtual environment (recommended)

```bash
git clone https://github.com/oslocyclotronlab/ompy.git
cd ompy
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
```

Install optional acceleration extras if you plan to run unfolding or other
JAX-backed routines:

```bash
pip install "jax[cpu]" jax-tqdm optax
```

## Option 2 — Docker workflow

```bash
git clone https://github.com/oslocyclotronlab/ompy.git
cd ompy
docker build -t ompy-dev .
docker run -it --rm -v $(pwd):/workspace ompy-dev bash
```

You now have an isolated environment with all build tools available. Use
`docker compose` if you prefer a long-lived container or want to share the setup
with collaborators.
