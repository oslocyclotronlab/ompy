# version info and git commit at execution time
# see also setup.py
from __future__ import annotations
from dataclasses import dataclass
import os
import subprocess
import pathlib
from .version_setup import version as VERSION


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env)
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except (subprocess.SubprocessError, OSError):
        GIT_REVISION = "Unknown"

    if not GIT_REVISION:
        # this shouldn't happen but apparently can (see gh-8512)
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of ompy.version messes up the build under Python 3.
    FULLVERSION = VERSION

    cwd = pathlib.Path.cwd()
    filepwd = pathlib.Path(__file__).parent.absolute()
    os.chdir(filepwd / "../")

    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('ompy/version_setup.py'):
        # must be a source distribution, use existing version file
        try:
            from .version_setup import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "ompy/version_setup.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    os.chdir(cwd)  # reset state

    return FULLVERSION, GIT_REVISION


FULLVERSION, GIT_REVISION = get_version_info()

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
    if '+' in version:
        head, tail = version.split('+')
        head = head.split('.dev')[0]
        return head, tail
    return version, ''

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

FULLVERSION_v = Version.from_str(FULLVERSION)
def warn_version(version: Version | str) -> None:
    if isinstance(version, str):
        version = Version.from_str(version)
    if version > FULLVERSION_v:
        raise RuntimeError(f"Version {version} is newer than {FULLVERSION_v}")

