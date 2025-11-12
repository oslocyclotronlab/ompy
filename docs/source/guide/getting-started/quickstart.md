# Quickstart

Follow this checklist after installation to confirm your environment works and
to explore the shipped examples.

1. Activate the environment (virtualenv or container) and ensure OMpy is on the
   `PYTHONPATH`.
2. Fetch example datasets and response functions as needed:
   ```bash
   ompy data fetch
   ```
   Downloads, verification, and extraction happen automatically under
   `~/.cache/ompy/`.
3. Launch Jupyter Lab with the bundled notebooks:
   ```bash
   jupyter lab examples/
   ```
   Each notebook fetches additional inputs on demand so you can follow along.
4. Inspect the CLI for available subcommands and enable shell completions:
   ```bash
   ompy --help
   register-python-argcomplete ompy  # add to shell profile for tab completion
   ```
5. Review the repository layout if you plan to contribute:
   - `src/ompy/` hosts the Python packages.
   - `examples/` gathers worked notebooks and scripts.
   - `tests/` runs the automated suite that backs CI.
