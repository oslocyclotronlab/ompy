from __future__ import annotations
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from importlib import metadata, util
from typing import Optional, List, Tuple

try:
    import colorama
    colorama.init(autoreset=True)
except ImportError:
    colorama = None

try:
    from IPython.display import display, HTML
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False

def colored(text: str, code: str) -> str:
    if colorama:
        color_map = {"green": colorama.Fore.GREEN, "red": colorama.Fore.RED, "yellow": colorama.Fore.YELLOW}
        return f"{color_map.get(code, '')}{text}{colorama.Style.RESET_ALL}"
    return f"\033[{code}m{text}\033[0m"

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
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free,memory.used",
            "--format=csv,nounits,noheader"
        ], encoding="utf-8")
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
        widths = [len(e.label) for e in self.entries] + [sub._text_width() for sub in self.subsections]
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
        details = '<details>' if self.collapsed else '<details open>'
        return f"{details}<summary><strong>{self.title}</strong>{status_html}</summary><div style='margin-left:1em'>{inner}</div></details>"
    def _repr_html_(self):
        return self.render_html()

def get_version_info() -> Tuple[str, str]:
    try:
        v = metadata.version("your-package-name")
    except metadata.PackageNotFoundError:
        v = "dev"
    sha = os.popen("git rev-parse --short HEAD").read().strip() or "n/a"
    return v, sha

@dataclass
class GPUMemory:
    total: int
    free: int
    used: int

def status() -> Section:
    root = Section("OMpy Status", collapsed=False)
    full_ver, git_sha = get_version_info()
    root.add_entry(InfoEntry("Version", full_ver))
    root.add_entry(InfoEntry("Git SHA", git_sha))
    from . import (
        GPU_AVAILABLE,
        NUMBA_AVAILABLE,
        NUMBA_CUDA_AVAILABLE,
        NUMBA_CUDA_WORKING,
        ROOT_AVAILABLE,
        ROOT_IMPORTED,
        UPROOT_AVAILABLE,
        JAX_AVAILABLE,
        JAX_WORKING,
        H5PY_AVAILABLE,
    )
    capability_specs = [
        ("GPU", GPU_AVAILABLE, []),
        ("NUMBA", NUMBA_AVAILABLE, [("CUDA available", NUMBA_CUDA_AVAILABLE),("CUDA working", NUMBA_CUDA_WORKING[0])]),
        ("ROOT", ROOT_AVAILABLE, [("imported", ROOT_IMPORTED)]),
        ("UPROOT", UPROOT_AVAILABLE, []),
        ("JAX", JAX_AVAILABLE, [("working", JAX_WORKING)]),
        ("h5py", H5PY_AVAILABLE, []),
    ]
    for label, avail, subs in capability_specs:
        sec = Section(label)
        sec.add_entry(StatusEntry("available", avail))
        for sub_label, sub_status in subs:
            sec.add_entry(StatusEntry(sub_label, sub_status))
        if label == "ROOT":
            if ROOT_IMPORTED:
                import ROOT
                ver = ROOT.__version__
            else:
                # This doesn't work, but a hail mary
                _, ver = is_available("ROOT")
        else:
            label = label.lower()
            _, ver = is_available(label)
        if ver:
            sec.add_entry(InfoEntry("version", ver))
        root.add_section(sec)
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
        plat.add_entry(InfoEntry("Total RAM (GB)", f"{mem.total/1024**3:.2f}"))
        plat.add_entry(InfoEntry("Avail RAM (GB)", f"{mem.available/1024**3:.2f}"))
    root.add_section(plat)
    if JAX_AVAILABLE:
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

def print_status():
    sec = get_status_section()
    print(sec)

