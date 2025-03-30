from __future__ import annotations
from dataclasses import dataclass

def split_version(version: str) -> tuple[int, int, int]:
    """
    Split a version string into a tuple of integers.

    >>> split_version('1.2.3')
    (1, 2, 3)
    """
    return tuple(int(v) for v in version.split('.'))

def split_git(version: str) -> tuple[str, str]:
    """
    Split a version string into a tuple of integers.

    >>> split_git('1.2.3.dev0+deadbeef')
    ('1.2.3', 'deadbeef')
    """
    head, tail = version.split('+')
    if tail:
        head = head.split('.dev')[0]
    return head, tail

def split(version: str) -> tuple[int, int, int, str]:
    """
    Split a version string into a tuple of integers.

    >>> split_version('1.2.3.dev0+deadbeef')
    (1, 2, 3, 'deadbeef')
    """
    head, tail = split_git(version)
    return (*split_version(head), tail)

def major(version: str) -> int:
    """
    Return the major version number.

    >>> major('1.2.3')
    1
    """
    return split(version)[0]

def minor(version: str) -> int:
    """
    Return the minor version number.

    >>> minor('1.2.3')
    2
    """
    return split(version)[1]

def micro(version: str) -> int:
    """
    Return the micro version number.

    >>> micro('1.2.3')
    3
    """
    return split(version)[2]


@dataclass
class Version:
    major: int
    minor: int
    micro: int
    git: str

    def is_dev(self) -> bool:
        return self.git != ''

    def __repr__(self) -> str:
        if self.is_dev():
            return f'{self.major}.{self.minor}.{self.micro}.dev0+{self.git}'
        return f'{self.major}.{self.minor}.{self.micro}'

    def __lt__(self, other: Version) -> bool:
        return (self.major, self.minor, self.micro) < (other.major, other.minor, other.micro)

    def __le__(self, other: Version) -> bool:
        return (self.major, self.minor, self.micro) <= (other.major, other.minor, other.micro)

    def __eq__(self, other: Version) -> bool:
        return (self.major, self.minor, self.micro) == (other.major, other.minor, other.micro)

    def __ne__(self, other: Version) -> bool:
        return (self.major, self.minor, self.micro) != (other.major, other.minor, other.micro)

    def __gt__(self, other: Version) -> bool:
        return (self.major, self.minor, self.micro) > (other.major, other.minor, other.micro)

    def __ge__(self, other: Version) -> bool:
        return (self.major, self.minor, self.micro) >= (other.major, other.minor, other.micro)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.micro, self.git))

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def from_str(cls, version: str) -> Version:
        return cls(*split(version))
