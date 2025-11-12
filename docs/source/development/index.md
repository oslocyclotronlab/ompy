# Development

This section gathers resources for contributors who want to work on OMpy
itself: running the test suite, understanding the architecture, updating the
documentation, and preparing releases. Fill in the subsections below as you
formalize your contributor workflow.

## Documentation workflow

The following steps mirror the tooling used in continuous integration for
building, linting, and validating the docs locally.

### Prerequisites

- Python 3.10 or newer.
- The project-specific virtual environment (for example `./ompy-dev`).
- Make (`make`) on Linux/macOS, or PowerShell on Windows.

Activate the environment before running any commands:

```bash
source ompy-dev/bin/activate
```

### Installing documentation dependencies

Install the Sphinx extras listed in `docs/requirements.txt`:

```bash
python -m pip install --upgrade -r docs/requirements.txt
```

The list covers the Furo theme, MyST parser, notebook support, and design
extensions used throughout the site.

### Building the documentation

```bash
make -C docs html
```

The rendered site is written to `docs/_build/html/index.html`. Open that file in
your browser to preview changes.

Run additional checks as needed:

```bash
make -C docs linkcheck
make -C docs clean
```

### Adding new content

1. Create the Markdown file under `docs/source/`. MyST directives, admonitions,
   and equations are supported.
2. Add the file to the relevant `toctree` (for example `docs/source/index.md`)
   so Sphinx includes it.
3. Rebuild the documentation (`make -C docs html`) and resolve warnings early.
