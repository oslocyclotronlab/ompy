from __future__ import annotations
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from importlib import metadata, util
from typing import Optional, List, Tuple, Union, Iterable, Dict, Any

from . import __version__ as _package_version
from .accel import (
    gpu_available,
    h5py_available,
    jax_available,
    jax_working,
    numba_available,
    numba_cuda_available,
    ROOT_imported,
    uproot_available,
)

try:
    import colorama

    colorama.init(autoreset=True)
except ImportError:
    colorama = None

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:
    get_ipython = None  # type: ignore[assignment]
    display = None  # type: ignore[assignment]


def _should_auto_display() -> bool:
    if not get_ipython or not display:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    if shell is None:
        return False
    return getattr(shell, "kernel", None) is not None


_AUTO_DISPLAY = _should_auto_display()

_ANSI_CODES = {"green": "32", "red": "31", "yellow": "33"}


def colored(text: str, code: str) -> str:
    if colorama:
        color_map = {
            "green": colorama.Fore.GREEN,
            "red": colorama.Fore.RED,
            "yellow": colorama.Fore.YELLOW,
        }
        return f"{color_map.get(code, '')}{text}{colorama.Style.RESET_ALL}"
    ansi = _ANSI_CODES.get(code)
    if ansi:
        return f"\033[{ansi}m{text}\033[0m"
    return text


def status_label(ok: Optional[bool]) -> str:
    if ok is True:
        return colored("OK", "green")
    if ok is False:
        return colored("NO", "red")
    return colored("Unknown", "yellow")


def is_available(pkg: str, import_it: bool = False) -> Tuple[bool, str]:
    try:
        spec = util.find_spec(pkg)
        if not spec:
            return False, ""
        version = metadata.version(pkg)
        if import_it:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                __import__(pkg)
        return True, version
    except Exception:
        return False, ""


def get_cpu_name() -> str:
    name = platform.processor()
    if name:
        return name
    try:
        sys_plat = platform.system()
        if sys_plat == "Linux":
            line = os.popen("grep 'model name' /proc/cpuinfo | uniq").read()
            return line.split(":", 1)[1].strip()
        if sys_plat == "Darwin":
            return os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
        if sys_plat == "Windows":
            out = os.popen("wmic cpu get name").read().strip().splitlines()
            return out[1].strip() if len(out) > 1 else ""
    except Exception:
        pass
    return "Unknown"


def get_gpu_memory() -> List[GPUMemory]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free,memory.used",
                "--format=csv,nounits,noheader",
            ],
            encoding="utf-8",
        )
        rows = [row.split(",") for row in out.strip().splitlines()]
        return [GPUMemory(int(t), int(f), int(u)) for t, f, u in rows]
    except Exception:
        return []


def cpu_count() -> int:
    return os.cpu_count() or 1


@dataclass
class Entry:
    label: str

    def render_text(self, width: int) -> str:
        raise NotImplementedError

    def render_html(self) -> str:
        raise NotImplementedError


@dataclass
class StatusEntry(Entry):
    ok: Optional[bool]

    def render_text(self, width: int) -> str:
        return f"{self.label.ljust(width)}: {status_label(self.ok)}"

    def render_html(self) -> str:
        color = {True: "green", False: "red", None: "orange"}[self.ok]
        txt = {True: "OK", False: "NO", None: "Unknown"}[self.ok]
        return f"<tr><td style='padding:2px 6px; text-align:left'>{self.label}</td><td style='padding:2px 6px; text-align:left; color:{color}'>{txt}</td></tr>"


@dataclass
class InfoEntry(Entry):
    info: str

    def render_text(self, width: int) -> str:
        return f"{self.label.ljust(width)}: {self.info}"

    def render_html(self) -> str:
        return f"<tr><td style='padding:2px 6px; text-align:left'>{self.label}</td><td style='padding:2px 6px; text-align:left'>{self.info}</td></tr>"


@dataclass
class Section:
    title: str
    entries: List[Entry] = field(default_factory=list)
    subsections: List[Section] = field(default_factory=list)
    collapsed: bool = True

    def add_entry(self, e: Entry) -> None:
        self.entries.append(e)

    def add_section(self, sec: Section) -> None:
        self.subsections.append(sec)

    def _text_width(self) -> int:
        widths = [len(e.label) for e in self.entries] + [
            sub._text_width() for sub in self.subsections
        ]
        return max(widths) if widths else 0

    def render_text(self, indent: int = 0) -> str:
        pad = self._text_width()
        indent_s = "  " * indent
        lines = [f"{indent_s}{self.title}", f"{indent_s}{'=' * len(self.title)}"]
        for e in self.entries:
            lines.append(f"{indent_s}{e.render_text(pad)}")
        for sub in self.subsections:
            lines.append(sub.render_text(indent + 1))
        return "\n".join(lines)

    def render_html(self) -> str:
        status_html = ""
        if self.entries and isinstance(self.entries[0], StatusEntry):
            ok = self.entries[0].ok
            color = {True: "green", False: "red", None: "orange"}[ok]
            txt = {True: "OK", False: "NO", None: "Unknown"}[ok]
            status_html = f" <span style='color:{color}'>({txt})</span>"
        rows = ["<table style='border-collapse:collapse;'>"]
        for e in self.entries:
            rows.append(e.render_html())
        rows.append("</table>")
        inner = "".join(rows)
        for sub in self.subsections:
            inner += sub.render_html()
        details = "<details>" if self.collapsed else "<details open>"
        return f"{details}<summary><strong>{self.title}</strong>{status_html}</summary><div style='margin-left:1em'>{inner}</div></details>"

    def _repr_html_(self):
        return self.render_html()

    def __str__(self) -> str:
        return self.render_text()

    def save(self, **kwargs) -> Path:
        return write_status(**kwargs)


def _normalize_key(label: str) -> str:
    cleaned = []
    for ch in label:
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in {" ", "-", "#", "/", ":"}:
            cleaned.append("_")
    key = "".join(cleaned).strip("_")
    return key or "item"


def _entries_to_mapping(entries: Iterable[Entry]) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    for entry in entries:
        base = _normalize_key(entry.label)
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        key = base if idx == 0 else f"{base}_{idx + 1}"
        if isinstance(entry, StatusEntry):
            val: Any
            if entry.ok is True:
                val = True
            elif entry.ok is False:
                val = False
            else:
                val = "unknown"
        elif isinstance(entry, InfoEntry):
            val = entry.info
        else:
            val = getattr(entry, "info", "")
        mapping[key] = val
    return mapping


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_quote(value)
    return _toml_quote(str(value))


def _section_to_toml(section: Section, path: List[str], lines: List[str]) -> None:
    table_name = ".".join(path)
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"[{table_name}]")
    lines.append(f"title = {_toml_quote(section.title)}")

    entries = _entries_to_mapping(section.entries)
    if entries:
        lines.append("")
        lines.append(f"[{table_name}.entries]")
        for key, value in entries.items():
            lines.append(f"{key} = {_toml_value(value)}")

    if section.subsections:
        sub_counts: Dict[str, int] = {}
        for sub in section.subsections:
            base = _normalize_key(sub.title)
            idx = sub_counts.get(base, 0)
            sub_counts[base] = idx + 1
            sub_key = base if idx == 0 else f"{base}_{idx + 1}"
            _section_to_toml(sub, path + ["subsections", sub_key], lines)


def section_to_toml(section: Section) -> str:
    """
    Render a Section hierarchy to TOML format.
    """
    lines: List[str] = []
    root_key = _normalize_key(section.title) or "status"
    _section_to_toml(section, [root_key], lines)
    return "\n".join(lines) + "\n"


def get_version_info() -> Tuple[str, str]:
    try:
        version = metadata.version("ompy")
    except metadata.PackageNotFoundError:
        version = _package_version or "dev"
    except Exception:
        version = _package_version or "dev"
    sha = os.popen("git rev-parse --short HEAD").read().strip() or "n/a"
    return version, sha


@dataclass
class GPUMemory:
    total: int
    free: int
    used: int


def get_status_section() -> Section:
    root = Section("OMpy Status", collapsed=False)
    full_ver, git_sha = get_version_info()
    root.add_entry(InfoEntry("Version", full_ver))
    root.add_entry(InfoEntry("Git SHA", git_sha))
    gpu_ok, gpu_msgs = gpu_available(verbose=True)
    numba_ok = numba_available()
    numba_cuda_ok, numba_cuda_msg = numba_cuda_available(verbose=True)
    root_imported = ROOT_imported()
    root_ok, root_version = is_available("ROOT")
    uproot_ok = uproot_available()
    _, uproot_version = is_available("uproot")
    jax_ok = jax_available()
    jax_work_ok, jax_work_msg = jax_working(verbose=True)
    _, jax_version = is_available("jax")
    h5_ok = h5py_available()
    _, h5_version = is_available("h5py")
    _, numba_version = is_available("numba")

    gpu_sec = Section("GPU")
    gpu_sec.add_entry(StatusEntry("available", gpu_ok))
    if isinstance(gpu_msgs, dict):
        for key, msg in gpu_msgs.items():
            if msg:
                label = key.replace("_", " ").title()
                gpu_sec.add_entry(InfoEntry(label, msg))
    root.add_section(gpu_sec)

    numba_sec = Section("NUMBA")
    numba_sec.add_entry(StatusEntry("available", numba_ok))
    numba_sec.add_entry(StatusEntry("CUDA runtime", numba_cuda_ok))
    if numba_cuda_msg:
        numba_sec.add_entry(InfoEntry("CUDA detail", numba_cuda_msg))
    if numba_version:
        numba_sec.add_entry(InfoEntry("version", numba_version))
    root.add_section(numba_sec)

    root_sec = Section("ROOT")
    root_sec.add_entry(StatusEntry("available", root_ok))
    root_sec.add_entry(StatusEntry("imported", root_imported))
    if root_imported:
        try:
            import ROOT  # type: ignore

            root_sec.add_entry(InfoEntry("version", ROOT.__version__))  # type: ignore[attr-defined]
        except Exception:
            if root_version:
                root_sec.add_entry(InfoEntry("version", root_version))
    elif root_version:
        root_sec.add_entry(InfoEntry("version", root_version))
    root.add_section(root_sec)

    uproot_sec = Section("UPROOT")
    uproot_sec.add_entry(StatusEntry("available", uproot_ok))
    if uproot_version:
        uproot_sec.add_entry(InfoEntry("version", uproot_version))
    root.add_section(uproot_sec)

    jax_sec = Section("JAX")
    jax_sec.add_entry(StatusEntry("available", jax_ok))
    jax_sec.add_entry(StatusEntry("working", jax_work_ok))
    if jax_work_msg:
        jax_sec.add_entry(InfoEntry("detail", jax_work_msg))
    if jax_version:
        jax_sec.add_entry(InfoEntry("version", jax_version))
    root.add_section(jax_sec)

    h5_sec = Section("h5py")
    h5_sec.add_entry(StatusEntry("available", h5_ok))
    if h5_version:
        h5_sec.add_entry(InfoEntry("version", h5_version))
    root.add_section(h5_sec)
    for pkg in ("xarray", "pymc", "pyro", "scikit-learn", "optax"):
        avail, ver = is_available(pkg)
        sec = Section(pkg)
        sec.add_entry(StatusEntry("installed", avail))
        if avail:
            sec.add_entry(InfoEntry("version", ver))
        root.add_section(sec)
    plat = Section("Platform")
    plat.add_entry(InfoEntry("OS", platform.platform()))
    plat.add_entry(InfoEntry("CPU", get_cpu_name()))
    plat.add_entry(InfoEntry("CPUs", str(cpu_count())))
    try:
        import psutil

        freq = psutil.cpu_freq().current
        mem = psutil.virtual_memory()
    except ImportError:
        freq = None
        mem = None
    if freq is not None:
        plat.add_entry(InfoEntry("CPU freq (MHz)", f"{freq:.2f}"))
    if mem is not None:
        plat.add_entry(InfoEntry("Total RAM (GB)", f"{mem.total / 1024**3:.2f}"))
        plat.add_entry(InfoEntry("Avail RAM (GB)", f"{mem.available / 1024**3:.2f}"))
    root.add_section(plat)
    if jax_ok:
        try:
            import jax
            import jaxlib

            jax_sec = Section("JAX Details")
            jax_sec.add_entry(InfoEntry("jax version", jax.__version__))
            jax_sec.add_entry(InfoEntry("jaxlib version", jaxlib.__version__))
            gmem = get_gpu_memory()
            for i, m in enumerate(gmem):
                g = Section(f"GPU#{i}")
                g.add_entry(InfoEntry("total MB", str(m.total)))
                g.add_entry(InfoEntry("free MB", str(m.free)))
                g.add_entry(InfoEntry("used MB", str(m.used)))
                jax_sec.add_section(g)
            root.add_section(jax_sec)
        except Exception:
            pass
    return root


def status() -> Section:
    """Build and optionally display the OMpy status overview."""
    sec = get_status_section()
    if _AUTO_DISPLAY:
        try:
            display(sec)
        except Exception:
            pass
    return sec


def write_status(
    path: Union[str, Path] = Path("ompy_status.toml"),
    *,
    html: bool = False,
    ensure_dir: bool = True,
) -> Path:
    """
    Persist the current OMpy status report to a file for later inspection.

    Args:
        path: Destination file path.
        html: When True write HTML markup, otherwise TOML.
        ensure_dir: Create parent directories when missing.

    Returns:
        The resolved `Path` pointing to the written file.
    """
    sec = get_status_section()
    out_path = Path(path)
    if ensure_dir:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if html:
        content = sec.render_html()
    else:
        content = section_to_toml(sec)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def print_status():
    sec = get_status_section()
    print(sec.render_text())
