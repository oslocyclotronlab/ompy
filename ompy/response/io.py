from ..stubs import Pathlike
from typing import Literal, TypeAlias, Iterable
from pathlib import Path
from .. import Vector
from collections import Counter
import numpy as np
from tqdm import tqdm
import pandas as pd
from os import scandir

ErrorHandling: TypeAlias = Literal['ignore', 'drop', 'raise', 'rebin']
Format = Literal['mama', 'ompy', 'numpy', 'feather', 'root']


def load(path: Pathlike, format: Format = 'mama', **kwargs) -> tuple[tuple[Vector, ...], tuple[Vector, np.ndarray]]:
    path = verify_path(path)
    match format:
        case 'mama':
            discrete = load_discrete_mama(path, **kwargs)
            compton = load_compton_mama(path, Eg=discrete[0].E, **kwargs)
            return discrete, compton
        case 'ompy' | 'numpy':
            return load_npy(path, **kwargs)
        case 'feather':
            raise NotImplementedError()
        case 'root':
            raise NotImplementedError()
        case _:
            raise ValueError(f"Unknown format {format}. Available formats are: {Format}")

def load_npy(path: Pathlike, compton_name: str = 'compton.npz', discrete_name: str = 'discrete.npy') -> tuple[tuple[Vector, ...], tuple[Vector, np.ndarray]]:
    path = Path(path)
    comptons = np.load(path / compton_name)
    discretes = np.load(path / discrete_name)
    fields = ('FE', 'SE', 'DE', 'AP', 'Eff')
    E = discretes['E']
    discrete = (Vector(E=E, values=discretes[field]) for field in fields)
    E_observed = comptons['E_observed']
    compton = (Vector(E=E_observed, values=c) for c in comptons.files if c != 'E_observed')
    # TODO Check for corruption
    return discrete, compton


def load_compton_mama(path: Pathlike, pattern: str = 'cmp*.m', Eg: Iterable[int] = None,
                      handle_error: ErrorHandling = 'raise') -> tuple[list[Vector], np.ndarray]:
    """ Load Compton response data from files.

    Parameters
    ----------
    path : Pathlike
        Path to directory containing the files.
    prefix : str, optional
        Prefix of the files, by default 'cmp'
    suffix : str, optional
        Suffix of the files, by default '.m'
    Eg : list[int], optional
        List of Eg values to load, by default None, in which case it loads all matching files.
    """
    path = Path(path)
    compton: list[Vector] = []
    # This energy is the true gamma energy. The compton vectors are indexed by the measured gamma energy
    E: list[int] = []
    prefix_len = len(pattern.split('*')[0])
    for file in path.glob(pattern):
        e = int(file.stem[prefix_len:])
        if Eg is not None:
            if e not in Eg:
                continue
        E.append(e)
        compton.append(Vector.from_path(file))
    if not len(compton):
        raise FileNotFoundError(f'No files found with glob pattern {pattern} in {path}')
    if Eg is not None and set(E) != set(Eg):
        raise FileNotFoundError(f'Not all files found. Missing: {set(Eg) - set(E)}')
    # Sort and check for equal calibration
    E, compton = zip(*sorted(zip(E, compton), key=lambda x: x[0]))
    E = list(E)
    compton = list(compton)
    calibrations = [(i, tuple(cmp.calibration().values())) for i, cmp in enumerate(compton)]
    counter = Counter([cal for i, cal in calibrations])
    if len(counter) > 1:
        if handle_error == 'raise':
            s = ''
            for elem, count in counter.items():
                s += f'{count} files with calibration {elem}\n'
                if count < 5:
                    s += "-------\n"
                    for i, cal in calibrations:
                        if cal == elem:
                            s += f'File E={E[i]} with calibration {cal}\n'
                    s += "========\n"
            raise ValueError(f'Not all files have the same calibration:\n{s}')
        elif handle_error == 'drop':
            common = counter.most_common(1)[0][0]
            to_drop = []
            for i, cal in calibrations:
                if cal != common:
                    to_drop.append(i)
            compton = [cmp for i, cmp in enumerate(compton) if i not in to_drop]
            E = [e for i, e in enumerate(E) if i not in to_drop]
        elif handle_error == 'rebin':
            # Rebin to largest binsize (rebinning smaller is impossible).
            largest = max({val for i, val in set(counter)})
            compton = [cmp.rebin(binwidth=largest) for cmp in tqdm(compton)]
        elif handle_error == 'ignore':
            pass
        else:
            raise ValueError(f'Unknown error handling {handle_error}')

    return compton, np.asarray(E)


def load_discrete_mama(path: Pathlike, name: str = 'resp.dat', read_fwhm: bool = True) -> tuple[Vector, Vector, Vector, Vector, Vector, Vector | None]:
    """ Load discrete response data from file.

    Parameters
    ----------
    path : Pathlike
        Path to file.
    name : str, optional
        Name of the file, by default 'resp.dat'
    read_fwhm : bool, optional
        Whether to read the FWHM, by default True. Only for
        backwards compatibility with old files.
    """
    path = Path(path)
    with open(path / name, 'r') as f:
        lines = f.readlines()

    number_of_lines = 0
    i = 0
    while (i := i+1) < len(lines):
        line = lines[i]
        # Number of lines. Some resps are misspelled
        if line.startswith("# Next: Num"):
            number_of_lines = int(lines[i+1])
            i += 1
            break

    df = pd.DataFrame([line.split() for line in lines[i+2:i+number_of_lines+3]],
                      columns=['E', 'FWHM', 'Eff', 'FE', 'SE', 'DE', 'AP'])
    df = df.astype(float)
    df['E'] = df['E'].astype(int)
    assert len(df) == number_of_lines, f"Corrupt {path / name}"
    E = df['E'].to_numpy()
    FE = Vector(E=E, values=df['FE'].to_numpy())
    DE = Vector(E=E, values=df['DE'].to_numpy())
    SE = Vector(E=E, values=df['SE'].to_numpy())
    AP = Vector(E=E, values=df['AP'].to_numpy())
    Eff = Vector(E=E, values=df['Eff'].to_numpy())
    FWHM = None
    if read_fwhm:
        FWHM = Vector(E=E, values=df['FWHM'].to_numpy())
    return FE, DE, SE, AP, Eff, FWHM


#### SAVING ####
# TODO Fix type annotations while avoiding circular imports

def save(path: Pathlike, data, format: Format = 'ompy', **kwargs) -> None:
    path = Path(path)
    match format:
        case 'mama':
            raise NotImplementedError()
        case 'ompy' | 'numpy':
            return save_npy(path, data, **kwargs)
        case 'feather':
            raise NotImplementedError()
        case 'root':
            raise NotImplementedError()
        case _:
            raise ValueError(f"Unknown format {format}. Available formats are: {Format}")


def save_npy(path: Pathlike, data, exists_ok: bool = False) -> None:
    path = Path(path)
    if not exists_ok and path.exists():
        raise FileExistsError(f'File {path} already exists.')
    mapping = {e: cmp for e, cmp in zip(data.E, data.compton)}
    mapping['E_observed'] = data.E_observed
    np.savez(path / 'compton.npz', **mapping)
    fields = ('FE', 'SE', 'DE', 'AP', 'Eff')
    arrays = (eval('data.'+name+'.values') for name in fields)
    E = data.E
    mapping = {k: v for k, v in zip(fields, arrays)}
    np.savez(path / 'discrete.npz', **mapping)

def convert_mama(output: Pathlike, path: Pathlike | None = None, data: Any | None = None, exist_ok: bool = False, **kwargs) -> None:
    """ Convert MAMA files to ompy format. """
    if (data is None and path is None) or (data is not None and path is not None):
        raise ValueError("Either path or data must be given.")
    if path is not None:
        from .responsedata import ResponseData
        path = verify_path(path)
        data = ResponseData.from_path(path, **kwargs)
    assert data is not None

    output = Path(output)
    output.mkdir(exist_ok=exist_ok, parents=True)
    save(output, data, format='ompy', exist_ok=exist_ok)

def verify_path(path: Pathlike) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Path {path} does not exist.')
    if not path.is_dir():
        raise ValueError(f'Path {path} is not a directory.')
    if not any(scandir(path)):
        raise ValueError(f'Path {path} is empty.')
    return path
