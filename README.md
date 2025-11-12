# OMpy — The Oslo Method in Python

OMpy is a Python library for analyzing nuclear data using the **Oslo Method**.  
It provides a complete workflow from coincidence matrices to unfolded spectra, first-generation extraction, and level-density and γ-strength-function analysis.  
The implementation emphasizes **reproducibility**, **statistical rigor**, and **extensibility**.

---

## ⚠️ Development status

This is a **developmental version** of OMpy (≥ 2.0.0).  
It **breaks compatibility** with all versions **earlier than 2.0.0**.  
Several internal modules, class names, and data paths have been redesigned for clarity and maintainability.

The package is **not yet published on PyPI**.  
Until release, you must install it manually using a **virtual environment** or **Docker** (instructions below).

---

## Installation (for now)

### Option 1 — Using a virtual environment (recommended)

These steps require a standard Python 3.12+ installation.

1. **Clone the repository**
   ```bash
   git clone --branch shapedev https://github.com/oslocyclotronlab/ompy.git
   cd ompy
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Upgrade packaging tools**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **Install OMpy in editable mode**
   ```bash
   pip install -e .
   ```

   This lets you edit the source files and immediately see changes.

---

### Option 2 — Using Docker

If you prefer isolation or do not want to install Python packages locally, you can build a containerized environment.

1. **Build the development image**
   ```bash
   docker build -t ompy-dev .
   ```

2. **Run an interactive shell**
   ```bash
   docker run -it --rm -v $(pwd):/workspace ompy-dev bash
   ```

   You can now test code, run notebooks, or build the package inside a clean environment.

3. **(Optional) Use docker-compose**

   Create a minimal `docker-compose.yml`:

   ```yaml
   services:
     dev:
       build: .
       volumes:
         - .:/workspace
       working_dir: /workspace
       command: bash
   ```

   Then start it:
   ```bash
   docker compose up dev
   ```

---

## Shell Completion

OMpy supports tab completion for bash, zsh, and fish shells. To enable it:

**For bash:**
```bash
# Add to ~/.bashrc
eval "$(register-python-argcomplete ompy)"
```

**For zsh:**
```bash
# Add to ~/.zshrc
autoload -U bashcompinit
bashcompinit
eval "$(register-python-argcomplete ompy)"
```

**For fish:**
```bash
# Add to ~/.config/fish/config.fish
register-python-argcomplete --shell fish ompy | source
```

After adding the appropriate line, restart your shell or source the config file:
```bash
source ~/.bashrc  # or ~/.zshrc for zsh
```

Now you can use tab completion with `ompy` commands:
```bash
ompy data <TAB>        # Shows: fetch, fetch-core, fetch-ripl, fetch-ensdf, verify, path, clean
ompy data fetch --<TAB>  # Shows available options
```

---

## Example data

OMpy does **not** ship large data files with the package.  
To download and unpack required datasets (response functions and RIPL-3 libraries), run:

```bash
ompy data fetch
```

This command will automatically download, verify, and extract all necessary files into a local cache directory such as:

```
~/.cache/ompy/
```

To remove the cache and start fresh:
```bash
ompy data clean
```

---

## Jupyter examples

Example notebooks demonstrating typical workflows are available in the `examples/` directory.

Start Jupyter Lab from the project root:
```bash
jupyter lab examples/
```

---

## Citation

If you use OMpy in scientific work, please cite:

> J. E. Midtbø, F. Zeiser, E. Lima  
> *A new software implementation of the Oslo method with rigorous statistical uncertainty propagation*,  
> *Computer Physics Communications* 261 (2021) 107795.  
> [DOI: 10.1016/j.cpc.2020.107795](https://doi.org/10.1016/j.cpc.2020.107795)

And include the original Oslo Method papers for unfolding and first-generation extraction.

---

## License

OMpy is open source and distributed under the MIT License.  
See [LICENSE.md](LICENSE.md) for details.
