from __future__ import annotations

import argparse
import hashlib
import os
import platform
import re
import shutil
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import zipfile

try:
    import argcomplete
    ARGCOMPLETE_AVAILABLE = True
except ImportError:
    ARGCOMPLETE_AVAILABLE = False


# =========================
# Version resolution
# =========================

def _pkg_version_release() -> str:
    """
    Return the release part (e.g., '2.1.0') of ompy.__version__.
    Falls back to '0.0.0' if unknown.
    """
    try:
        import ompy  # local import
        v = getattr(ompy, "__version__", "0+unknown")
    except Exception:
        v = "0+unknown"
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(v))
    return m.group(0) if m else "0.0.0"


def _series(version_release: str) -> str:
    """
    Take '2.1.0' -> '2.1' for series mapping.
    """
    m = re.match(r"^(\d+)\.(\d+)", version_release)
    return f"{m.group(1)}.{m.group(2)}" if m else "0.0"


# Map library series -> dataset version.
# Update this as you cut new releases.
DATA_VERSION_BY_SERIES = {
    "2.1": "v2.1.0-alpha",   # ompy 2.1.x -> dataset 2025.10
}

DATA_VERSION_DEFAULT = "v2.1.0-alpha"  # fallback

def resolve_data_version(cli_arg: str | None) -> str:
    """
    Priority:
      1) --data-version CLI flag
      2) OMPY_DATA_VERSION env var
      3) Series mapping from ompy.__version__
      4) DATA_VERSION_DEFAULT
    """
    if cli_arg and cli_arg.strip():
        return cli_arg.strip()
    env = os.environ.get("OMPY_DATA_VERSION")
    if env and env.strip():
        return env.strip()
    series = _series(_pkg_version_release())
    return DATA_VERSION_BY_SERIES.get(series, DATA_VERSION_DEFAULT)


# =========================
# Config (URL templates)
# =========================

CORE_URL_TEMPLATES = [
    "https://github.com/oslocyclotronlab/ompy/releases/download/{tag}/response.tar.gz",
]

CORE_SHA256_BY_VERSION = {
    "v2.1.0-alpha": "ed6489b3b514142d4548a616ba4a105467d7eb82a0d730473fe1c51f587323f3",
}

# --- RIPL-3 items (download individually; stdlib-only) ---
RIPL_ITEMS = [
    {
        "url": "https://www-nds.iaea.org/RIPL-3/levels/levels.zip",
        "dest": "ripl3/levels.zip",
        "install": "ripl3",
        "kind": "zip",
        "sha256": ""
    },
    {
        "url": "https://www-nds.iaea.org/RIPL-3/densities/level-densities-bfmeff.dat",
        "dest": "ripl3/densities/level-densities-bfmeff.dat",
        "install": "ripl3/densities/level-densities-bfmeff.dat",
        "kind": "file",
        "sha256": "7ff7186b6b069095fe66e90bf5f686f88b46e1017114bd4e0e04012f6f2116dc"
    },
    {
        "url": "https://www-nds.iaea.org/RIPL-3/densities/level-densities-ctmeff.dat",
        "dest": "ripl3/densities/level-densities-ctmeff.dat",
        "install": "ripl3/densities/level-densities-ctmeff.dat",
        "kind": "file",
        "sha256": "8197e2b4c564f9dc9b7defc20b829fa2e96a05b301d0a7a1beb94f797b4d0904"
    },
    {
        "url": "https://www-nds.iaea.org/RIPL-3/resonances/resonances0.dat",
        "dest": "ripl3/resonances/resonances0.dat",
        "install": "ripl3/resonances/resonances0.dat",
        "kind": "file",
        "sha256": ""
    },
    {
        "url": "https://www-nds.iaea.org/RIPL-3/resonances/resonances1.dat",
        "dest": "ripl3/resonances/resonances1.dat",
        "install": "ripl3/resonances/resonances1.dat",
        "kind": "file",
        "sha256": ""
    },
]
RIPL_EXTRACT_DIR = "ripl3"  # root folder for all RIPL content

# --- ENSDF (nudel) ---
# Default URL (warn users it's likely outdated); user can override via CLI flags.
ENSDF_DEFAULT_URL = "https://www.nndc.bnl.gov/ensdfarchivals/distributions/dist25/ensdf_251001.zip"
ENSDF_INSTALL_SUBDIR = "ensdf"  # under our cache where we actually store the files


@dataclass
class DataConfig:
    data_version: str
    cache: Path

    @property
    def core_archive_name(self) -> str:
        return f"response-{self.data_version}.tar.gz"

    @property
    def core_extract_dirname(self) -> str:
        return f"response-{self.data_version}"

    @property
    def core_urls(self) -> list[str]:
        return [u.format(tag=self.data_version) for u in CORE_URL_TEMPLATES]

    @property
    def core_sha256(self) -> str | None:
        return CORE_SHA256_BY_VERSION.get(self.data_version)


# =========================
# Helpers
# =========================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_item(item: dict, cache: Path) -> Path:
    dest = cache / item["dest"]
    ensure_parent(dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    download_with_retries([item["url"]], tmp)
    if item.get("sha256"):
        verify_sha256(tmp, item["sha256"])  # will raise on mismatch
    # Ensure parent exists before rename (in case it was cleaned up)
    ensure_parent(dest)
    shutil.move(str(tmp), str(dest))
    return dest


def install_item(item: dict, cache: Path, downloaded: Path) -> None:
    install_target = cache / item["install"]
    ensure_parent(install_target)

    kind = item.get("kind", "file").lower()
    if kind == "zip":
        target_dir = install_target
        # Only remove if target_dir exists and downloaded file is not inside it
        if target_dir.exists():
            try:
                # Check if downloaded file is inside target_dir
                downloaded.relative_to(target_dir)
                is_inside = True
            except ValueError:
                is_inside = False
            
            if is_inside:
                # Remove only the contents we'll be replacing, not the downloaded zip
                # For now, just skip the removal - the extraction will overwrite
                pass
            else:
                shutil.rmtree(target_dir)
        
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(target_dir)
        print(f"✓ Extracted ZIP to {target_dir}")

    elif kind == "file":
        if downloaded.resolve() == install_target.resolve():
            print(f"✓ File already in place: {install_target}")
            return
        shutil.copy2(downloaded, install_target)
        print(f"✓ Installed file to {install_target}")

    else:
        raise ValueError(f"Unknown RIPL item kind: {kind}")


def default_cache_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "ompy"
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", str(Path.home()))
        return Path(base) / "ompy" / "Cache"
    return Path.home() / ".cache" / "ompy"


def get_cache_dir(cli_arg: str | None) -> Path:
    return Path(
        cli_arg
        or os.environ.get("OMPY_DATA_DIR")
        or default_cache_dir()
    ).expanduser().resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024


def download_with_retries(urls: list[str], dest: Path, retries: int = 3) -> None:
    last_err = None
    for url in urls:
        if not url:
            continue
        for attempt in range(1, retries + 1):
            try:
                print(f"→ Downloading: {url}  (attempt {attempt}/{retries})")
                _download(url, dest)
                size = dest.stat().st_size if dest.exists() else 0
                print(f"✓ Saved to {dest} ({human(size)})")
                return
            except (URLError, HTTPError, OSError) as e:
                last_err = e
                print(f"  ! Failed: {e}  (attempt {attempt})")
                time.sleep(1.5 * attempt)
        print("  ! URL failed after retries, trying next mirror...")
    raise RuntimeError(f"All downloads failed. Last error: {last_err}")


def _download(url: str, dest: Path) -> None:
    req = Request(url, headers={"User-Agent": "ompy/cli"})
    with urlopen(req) as resp, dest.open("wb") as out:
        length = resp.headers.get("Content-Length")
        total = int(length) if length and str(length).isdigit() else None
        read = 0
        block = 1024 * 64
        while True:
            chunk = resp.read(block)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if total:
                pct = read * 100 // total
                print(f"\r   {human(read)} / {human(total)} ({pct}%)", end="", flush=True)
        print("")


def verify_sha256(path: Path, expected: str | None) -> None:
    if not expected or expected.strip("<> ") == "":
        print("! Skipping SHA256 verification (no expected hash configured).")
        return
    got = sha256_file(path)
    if got.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path.name}:\n  expected: {expected}\n  got:      {got}")
    print(f"✓ SHA256 OK ({got})")


def extract_tar_gz(archive: Path, target_dir: Path) -> None:
    print(f"→ Extracting {archive.name} …")
    with tarfile.open(archive, "r:gz") as tf:
        top_members = {m.name.split("/")[0] for m in tf.getmembers() if m.name and "/" in m.name}
        has_single_top = len(top_members) == 1
        tmp = target_dir.parent / (target_dir.name + ".__extracting__")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        tf.extractall(tmp)
        if has_single_top:
            only = next(iter(top_members))
            inner = tmp / only
            if inner.exists():
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                inner.rename(target_dir)
                shutil.rmtree(tmp, ignore_errors=True)
                print(f"✓ Extracted to {target_dir}")
                return
        if target_dir.exists():
            shutil.rmtree(target_dir)
        tmp.rename(target_dir)
    print(f"✓ Extracted to {target_dir}")


# Pretty print for verify

def _print_hash_report(title: str, rows: list[tuple[str, str, str]]):
    if not rows:
        return
    print(f"\n{title}")
    print("-" * max(24, len(title)))
    col1 = max(len(r[0]) for r in rows) if rows else 10
    col2 = 64  # sha256 length
    for label, actual, status in rows:
        print(f"{label.ljust(col1)}  {actual or '-':<{col2}}  {status}")


# =========================
# ENSDF (nudel) helpers
# =========================

def _ensdf_target_dir(cfg: DataConfig) -> Path:
    return cfg.cache / ENSDF_INSTALL_SUBDIR


def _detect_ensdf_present(dirpath: Path) -> bool:
    if not dirpath.exists() or not dirpath.is_dir():
        return False
    for p in dirpath.iterdir():
        if p.is_file() and p.name.startswith("ensdf."):
            return True
    return False


def _extract_ensdf_archive(archive: Path, target_dir: Path) -> None:
    print(f"→ Extracting ENSDF archive {archive.name} …")
    try:
        with zipfile.ZipFile(archive) as zf:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(target_dir)
            print(f"✓ ENSDF extracted to {target_dir}")
            return
    except zipfile.BadZipFile:
        pass
    try:
        with tarfile.open(archive, mode="r:gz") as tf:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            tf.extractall(target_dir)
            print(f"✓ ENSDF extracted to {target_dir}")
            return
    except (tarfile.ReadError, tarfile.CompressionError):
        pass
    raise RuntimeError(f"Unrecognized ENSDF archive format: {archive}")


def _xdg_data_home() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg).expanduser()
    return Path.home() / ".local" / "share"


def _nudel_expected_path() -> Path:
    env = os.environ.get("ENSDF_PATH")
    if env and env.strip():
        return Path(env).expanduser()
    return _xdg_data_home() / "ensdf"


def _safe_symlink(src: Path, dest: Path) -> None:
    dest_parent = dest.parent
    dest_parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            if dest.is_symlink() and dest.resolve() == src.resolve():
                print(f"✓ ENSDF symlink already correct: {dest} -> {src}")
                return
        except OSError:
            pass
        if dest.is_symlink():
            dest.unlink()
        elif dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()
        else:
            print(f"! ENSDF destination exists and is not empty: {dest}\n  Not replacing it. Please adjust manually if needed.")
            return
    try:
        os.symlink(src, dest, target_is_directory=True)
        print(f"✓ Created ENSDF symlink: {dest} -> {src}")
    except OSError as e:
        print(f"! Could not create symlink ({e}). As a fallback, set ENSDF_PATH={src}")


# =========================
# Actions (parameterized by DataConfig)
# =========================

def action_fetch_core(cfg: DataConfig, force: bool = False) -> None:
    cfg.cache.mkdir(parents=True, exist_ok=True)
    archive = cfg.cache / cfg.core_archive_name
    outdir  = cfg.cache / cfg.core_extract_dirname

    if outdir.exists() and not force:
        print(f"✓ Core data already present: {outdir}")
        return

    if not archive.exists() or force:
        if archive.exists():
            archive.unlink()
        download_with_retries(cfg.core_urls, archive)

    if cfg.core_sha256:
        verify_sha256(archive, cfg.core_sha256)
    else:
        print("! No expected SHA for core configured; skipping strict check.")
    extract_tar_gz(archive, outdir)


def action_fetch_ripl(cfg: DataConfig, force: bool = False) -> None:
    root = cfg.cache / RIPL_EXTRACT_DIR
    root.mkdir(parents=True, exist_ok=True)

    all_present = True
    for it in RIPL_ITEMS:
        target = cfg.cache / it["install"]
        if it.get("kind", "file") == "zip":
            if not (target.exists() and target.is_dir()):
                all_present = False
                break
        else:
            if not target.exists():
                all_present = False
                break

    if all_present and not force:
        print(f"✓ RIPL3 already present under: {root}")
        return

    for it in RIPL_ITEMS:
        print(f"→ RIPL: {it['url']}")
        downloaded = download_item(it, cfg.cache)
        install_item(it, cfg.cache, downloaded)

    print(f"✓ RIPL3 ready at: {root}")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def action_fetch_ensdf_for_nudel(cfg: DataConfig, *, ensdf_url: str | None = None, ensdf_zip: str | None = None, force: bool = False) -> None:
    """Prepare ENSDF for nudel and create the expected symlink.

    Storage layout:
      - Real files live under:  <cache>/ensdf/
      - We create a symlink at: $ENSDF_PATH or $XDG_DATA_HOME/ensdf  -> <cache>/ensdf
    """
    target_dir = _ensdf_target_dir(cfg)
    source = ensdf_url or ensdf_zip
    if not source:
        print(
            "! Using default ENSDF URL (likely outdated):\n"
            f"  {ENSDF_DEFAULT_URL}\n"
            "  Find the latest at: https://www.nndc.bnl.gov/ensdfarchivals/\n"
            "  (override via --ensdf-url URL or --ensdf-zip /path/to/zip)."
        )
        source = ENSDF_DEFAULT_URL

    if _detect_ensdf_present(target_dir) and not force:
        print(f"✓ ENSDF already present under: {target_dir}")
    else:
        if _is_url(source):
            archive_path = cfg.cache / "ensdf_archive.part"
            download_with_retries([source], archive_path)
            expected = os.environ.get("OMPY_ENSDF_SHA256", "").strip()
            if expected:
                verify_sha256(archive_path, expected)
            _extract_ensdf_archive(archive_path, target_dir)
            final_archive = cfg.cache / "ensdf_archive"
            if final_archive.exists():
                final_archive.unlink()
            archive_path.rename(final_archive)
        else:
            src_path = Path(source).expanduser().resolve()
            if not src_path.exists():
                raise FileNotFoundError(f"ENSDF source not found: {src_path}")
            _extract_ensdf_archive(src_path, target_dir)

    dest = _nudel_expected_path()
    _safe_symlink(target_dir, dest)


def action_fetch(cfg: DataConfig, *, ensdf_url: str | None = None, ensdf_zip: str | None = None, force: bool = False) -> None:
    print(f"Cache: {cfg.cache}  |  Dataset: {cfg.data_version}")
    action_fetch_core(cfg, force=force)
    action_fetch_ripl(cfg, force=force)
    action_fetch_ensdf_for_nudel(cfg, ensdf_url=ensdf_url, ensdf_zip=ensdf_zip, force=force)
    print("\n✓ OMpy data setup complete.")
    print(f"  Location: {cfg.cache}")
    print("  Tip: set OMPY_DATA_DIR to change the cache directory.")
    print("  Tip: set OMPY_DATA_VERSION or use --data-version to pin dataset version.")
    print("  Tip: provide ENSDF via --ensdf-url or --ensdf-zip; otherwise a default (possibly outdated) URL is used.")


def action_verify(cfg: DataConfig) -> int:
    status_ok = True

    core_rows: list[tuple[str, str, str]] = []
    archive = cfg.cache / cfg.core_archive_name
    if archive.exists():
        actual = sha256_file(archive)
        if cfg.core_sha256 and cfg.core_sha256.strip("<> "):
            if actual.lower() == cfg.core_sha256.lower():
                status = "OK (matches expected)"
            else:
                status = "MISMATCH"
                status_ok = False
        else:
            status = "no expected configured"
        core_rows.append((archive.name, actual, status))
    else:
        core_rows.append((archive.name, "-", "missing"))
        status_ok = False

    core_dir = cfg.cache / cfg.core_extract_dirname
    core_rows.append((f"{core_dir.name}/", "-", "present" if core_dir.exists() else "missing"))
    if not core_dir.exists():
        status_ok = False

    _print_hash_report("Core dataset", core_rows)

    ripl_rows: list[tuple[str, str, str]] = []
    for it in RIPL_ITEMS:
        kind = it.get("kind", "file")
        hash_path = (cfg.cache / it["dest"]) if kind == "zip" else (cfg.cache / it["install"])
        label = it["dest"] if kind == "zip" else it["install"]
        expected = (it.get("sha256") or "").strip()
        if hash_path.exists():
            actual = sha256_file(hash_path)
            if expected:
                if actual.lower() == expected.lower():
                    status = "OK (matches expected)"
                else:
                    status = "MISMATCH"
                    status_ok = False
            else:
                status = "no expected configured"
            ripl_rows.append((label, actual, status))
        else:
            ripl_rows.append((label, "-", "missing"))
            status_ok = False

    levels_dir = cfg.cache / "ripl3/levels"
    ripl_rows.append(("ripl3/levels/", "-", "present" if levels_dir.exists() else "missing"))

    _print_hash_report("RIPL-3 items", ripl_rows)

    ensdf_rows: list[tuple[str, str, str]] = []
    ensdf_dir = _ensdf_target_dir(cfg)
    ensdf_rows.append((str(ensdf_dir) + "/", "-", "present" if _detect_ensdf_present(ensdf_dir) else "missing"))
    expected_dest = _nudel_expected_path()
    link_status = "missing"
    try:
        if expected_dest.is_symlink() and expected_dest.resolve() == ensdf_dir.resolve():
            link_status = "symlink OK"
        elif expected_dest.exists():
            link_status = "exists (not our symlink)"
        else:
            link_status = "missing"
    except OSError:
        link_status = "unreadable"
    ensdf_rows.append((str(expected_dest), "-", link_status))
    _print_hash_report("ENSDF (nudel)", ensdf_rows)

    if status_ok and _detect_ensdf_present(ensdf_dir):
        print("\n✓ Data presence & hashes look OK.")
        return 0
    else:
        print("\n! Some data are missing or have hash mismatches.")
        print("  If you see 'no expected configured', please copy the shown SHA256 and report it to the developers.")
        return 2


def action_show_path(cfg: DataConfig) -> None:
    print(f"\nOMpy data information")
    print(f"─────────────────────────────")
    print(f"Dataset version : {cfg.data_version}")
    print(f"Cache directory : {cfg.cache}")
    core_dir = cfg.cache / cfg.core_extract_dirname
    ripl_dir = cfg.cache / "ripl3"
    ensdf_dir = _ensdf_target_dir(cfg)
    nudel_dest = _nudel_expected_path()
    print(f"Core data       : {core_dir}{' ✅' if core_dir.exists() else ' (missing)'}")
    print(f"RIPL3 data      : {ripl_dir}{' ✅' if ripl_dir.exists() else ' (missing)'}")
    print(f"ENSDF (nudel)   : {ensdf_dir}{' ✅' if _detect_ensdf_present(ensdf_dir) else ' (missing)'}")
    print(f"nudel link      : {nudel_dest}{' ✅' if (nudel_dest.is_symlink() and nudel_dest.resolve()==ensdf_dir.resolve()) else ''}")
    print()
    print("Tip: set OMPY_DATA_DIR or use --cache to override.")
    print("Tip: set OMPY_DATA_VERSION or use --data-version to pin dataset version.")
    print("Tip: use --ensdf-url or --ensdf-zip to control ENSDF acquisition.")


def action_clean(cfg: DataConfig) -> None:
    """Remove the entire data cache directory."""
    cache_path = cfg.cache
    
    if not cache_path.exists():
        print(f"✓ Cache directory does not exist: {cache_path}")
        return
    
    print(f"Removing cache directory: {cache_path}")
    shutil.rmtree(cache_path)
    print(f"✓ Cache directory removed: {cache_path}")


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ompy", description="OMpy command-line tools")
    sub = p.add_subparsers(dest="cmd", metavar="command")

    p_data = sub.add_parser("data", help="Data management")
    subd = p_data.add_subparsers(dest="data_cmd", metavar="subcommand")

    def add_shared(a: argparse.ArgumentParser) -> None:
        a.add_argument("--cache", help="Cache directory (default: OMPY_DATA_DIR or platform default)")
        a.add_argument("--data-version", help="Dataset version (default: derived from ompy version or OMPY_DATA_VERSION)")

    q = subd.add_parser("fetch", help="Fetch & prepare required data (core + RIPL3 + ENSDF)")
    add_shared(q)
    q.add_argument("--force", action="store_true", help="Re-download/re-extract even if present")
    q.add_argument("--ensdf-url", help="ENSDF archive URL (ZIP/TGZ)")
    q.add_argument("--ensdf-zip", help="Path to local ENSDF ZIP/TGZ")

    fc = subd.add_parser("fetch-core", help="Fetch core tarball only")
    add_shared(fc)
    fc.add_argument("--force", action="store_true", help="Re-download/re-extract even if present")

    fr = subd.add_parser("fetch-ripl", help="Fetch RIPL3 only")
    add_shared(fr)
    fr.add_argument("--force", action="store_true", help="Re-download/re-extract even if present")

    fe = subd.add_parser("fetch-ensdf", help="Fetch ENSDF and create the expected symlink")
    add_shared(fe)
    fe.add_argument("--force", action="store_true", help="Re-download/re-extract even if present")
    fe.add_argument("--ensdf-url", help="ENSDF archive URL (ZIP/TGZ)")
    fe.add_argument("--ensdf-zip", help="Path to local ENSDF ZIP/TGZ")

    v = subd.add_parser("verify", help="Verify presence and print SHA256 for all managed files")
    add_shared(v)

    pth = subd.add_parser("path", help="Show current data cache and dataset version")
    add_shared(pth)

    c = subd.add_parser("clean", help="Remove the entire data cache directory")
    add_shared(c)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    
    # Enable shell completion if argcomplete is available and we're in a real shell
    if ARGCOMPLETE_AVAILABLE and argv is None:
        argcomplete.autocomplete(parser)
    
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)

    if args.cmd != "data" or args.data_cmd is None:
        parser.print_help(file=sys.stderr)
        return 2

    cache = get_cache_dir(getattr(args, "cache", None))
    data_version = resolve_data_version(getattr(args, "data_version", None))
    cfg = DataConfig(data_version=data_version, cache=cache)

    try:
        if args.data_cmd == "fetch":
            action_fetch(cfg, ensdf_url=getattr(args, "ensdf_url", None), ensdf_zip=getattr(args, "ensdf_zip", None), force=args.force)
            return 0
        if args.data_cmd == "fetch-core":
            action_fetch_core(cfg, force=args.force)
            return 0
        if args.data_cmd == "fetch-ripl":
            action_fetch_ripl(cfg, force=args.force)
            return 0
        if args.data_cmd == "fetch-ensdf":
            action_fetch_ensdf_for_nudel(cfg, ensdf_url=getattr(args, "ensdf_url", None), ensdf_zip=getattr(args, "ensdf_zip", None), force=args.force)
            return 0
        if args.data_cmd == "verify":
            return action_verify(cfg)
        if args.data_cmd == "path":
            action_show_path(cfg)
            return 0
        if args.data_cmd == "clean":
            action_clean(cfg)
            return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    parser.print_help(file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
