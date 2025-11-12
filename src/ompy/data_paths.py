from __future__ import annotations
import os
import platform
from pathlib import Path

def default_cache_dir() -> Path:
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "ompy"
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", str(Path.home()))
        return Path(base) / "ompy" / "Cache"
    return Path.home() / ".cache" / "ompy"

def data_root() -> Path:
    return Path(os.environ.get("OMPY_DATA_DIR", default_cache_dir()))

def data_exists() -> bool:
    return data_root().exists()

def get_response_path(version: str = 'latest') -> Path:
    """Return path to response data directory.
    
    Args:
        version: Response version to use. If 'latest', will use latest available version.
    
    Returns:
        Path to response data directory
    """
    if not data_exists():
        raise FileNotFoundError(f"Data root {data_root()} does not exist. "
                                "Please set up using `ompy data fetch` or set the OMPY_DATA_DIR environment variable."
                                f"OMPY_DATA_DIR is set to {os.environ.get('OMPY_DATA_DIR')}")
    root = data_root()
    paths = [p for p in root.glob("response-*") if p.is_dir()]
    if not paths:
        raise FileNotFoundError(f"No response data found in {root}. Please set up the response data using `ompy data fetch-core`.")
    
    if version == 'latest':
        # Sort by version number to get latest
        paths.sort(reverse=True)
        return paths[0]
        
    # Look for specific version
    version_path = root / f"response-{version}"
    if not version_path.exists() or not version_path.is_dir():
        raise FileNotFoundError(f"Response version {version} not found in {root}. Available versions: {', '.join([p.name.replace('response-', '') for p in paths])}")
    return version_path