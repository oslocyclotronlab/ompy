from __future__ import annotations

from dataclasses import dataclass

try:  # Prefer the setuptools-scm generated version when available.
    from ._version import version as FULLVERSION  # type: ignore
except Exception:  # pragma: no cover - fallback during editable installs
    FULLVERSION = "0+unknown"


def _parse_release_component(component: str) -> int:
    """Extract the leading integer value from a dot-separated version field."""
    digits = []
    for char in component:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return int("".join(digits)) if digits else 0


def _split_version(version: str) -> tuple[int, int, int, str]:
    """Split a PEP 440-ish version into major/minor/micro and optional git hash."""
    if not version:
        return 0, 0, 0, ""

    head = version
    git = ""

    if "+" in version:
        head, local = version.split("+", 1)
        git = local.split(".", 1)[0] or ""

    if ".post" in head:
        head = head.split(".post", 1)[0]
    if ".dev" in head:
        head = head.split(".dev", 1)[0]

    parts = head.split(".")
    major = _parse_release_component(parts[0]) if len(parts) > 0 else 0
    minor = _parse_release_component(parts[1]) if len(parts) > 1 else 0
    micro = _parse_release_component(parts[2]) if len(parts) > 2 else 0

    return major, minor, micro, git


def _coerce_version(value: Version | str) -> Version:
    if isinstance(value, Version):
        return value
    if isinstance(value, str):
        return Version.from_str(value)
    raise TypeError(f"Cannot compare Version with {type(value)!r}")


@dataclass(frozen=True)
class Version:
    """Minimal semantic version helper compatible with legacy behaviour."""

    major: int
    minor: int
    micro: int
    git: str = ""
    raw: str = ""

    def __post_init__(self) -> None:
        if not self.raw:
            object.__setattr__(self, "raw", self._render())

    def _render(self) -> str:
        base = f"{self.major}.{self.minor}.{self.micro}"
        return f"{base}.dev0+{self.git}" if self.git else base

    def is_dev(self) -> bool:
        return bool(self.git)

    def __repr__(self) -> str:  # pragma: no cover - repr delegates to __str__
        return str(self)

    def __str__(self) -> str:
        return self.raw or self._render()

    def _cmp_key(self) -> tuple[int, int, int]:
        return self.major, self.minor, self.micro

    def __lt__(self, other: Version | str) -> bool:
        return self._cmp_key() < _coerce_version(other)._cmp_key()

    def __le__(self, other: Version | str) -> bool:
        return self._cmp_key() <= _coerce_version(other)._cmp_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Version, str)):
            return NotImplemented
        return self._cmp_key() == _coerce_version(other)._cmp_key()

    def __ne__(self, other: object) -> bool:  # pragma: no cover - derived from __eq__
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __gt__(self, other: Version | str) -> bool:
        return self._cmp_key() > _coerce_version(other)._cmp_key()

    def __ge__(self, other: Version | str) -> bool:
        return self._cmp_key() >= _coerce_version(other)._cmp_key()

    def __hash__(self) -> int:  # pragma: no cover - dataclass default OK
        return hash((self.major, self.minor, self.micro, self.git))

    @classmethod
    def from_str(cls, version: str) -> Version:
        major, minor, micro, git = _split_version(version)
        return cls(major=major, minor=minor, micro=micro, git=git, raw=version)


CURRENT_VERSION = Version.from_str(FULLVERSION)
GIT_REVISION = CURRENT_VERSION.git or "Unknown"


def warn_version(version: Version | str) -> None:
    if CURRENT_VERSION.raw == "0+unknown":
        return
    candidate = _coerce_version(version)
    if candidate > CURRENT_VERSION:
        raise RuntimeError(f"Version {candidate} is newer than {CURRENT_VERSION}")


__all__ = ["Version", "warn_version", "FULLVERSION", "GIT_REVISION"]
