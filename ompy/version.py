# version info and git commit at execution time
# see also setup.py
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
    elif os.path.exists('ompy/version.py'):
        # must be a source distribution, use existing version file
        try:
            from ompy.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "ompy/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    os.chdir(cwd)  # reset state

    return FULLVERSION, GIT_REVISION


FULLVERSION, GIT_REVISION = get_version_info()
