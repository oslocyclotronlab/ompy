from ..stubs import Pathlike
from typing import Literal, TypeAlias, Iterable, Any
from pathlib import Path
from .. import Vector, Matrix
from collections import Counter
import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
from os import scandir

ErrorHandling: TypeAlias = Literal['ignore', 'drop', 'raise', 'rebin']
Format = Literal['mama', 'ompy', 'numpy', 'feather', 'root']

# TODO: the ompy format should save vectors correctly. Per now the metadata is discarded.

def load(path: Pathlike, format: Format = 'mama', **kwargs) -> tuple[tuple[Vector, ...], Matrix]:
    path = verify_path(path)
    match format:
        case 'mama':
            discrete = load_discrete_mama(path)
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


def load_npy(path: Pathlike, compton_name: str = 'compton.npz', discrete_name: str = 'discrete.npz') -> tuple[tuple[Vector, ...], Matrix]:
    path = Path(path)
    with np.load(path / compton_name) as comptons:
        E_observed = comptons['E_observed']
        E_true = comptons['E_true']
        values = comptons['values']
    with np.load(path / discrete_name) as discretes:
        fields = ('FE', 'SE', 'DE', 'AP', 'Eff')
        E = discretes['E']
        discrete = tuple(Vector(E=E, values=discretes[field], name=field) for field in fields)
    compton = Matrix(Ex=E_true, Eg=E_observed, values=values)
    if not np.allclose(E_true, E):
        raise RuntimeError("E_true != E. Probably save corruption.")
    return discrete, compton


def load_compton_mama(path: Pathlike, pattern: str = 'cmp*.m', Eg: Iterable[int] = None,
                      handle_error: ErrorHandling = 'raise') -> Matrix:
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
    Eg_: list[int] = []
    prefix_len = len(pattern.split('*')[0])
    for file in tqdm(path.glob(pattern)):
        e = int(file.stem[prefix_len:])
        if Eg is not None:
            if e not in Eg:
                continue
        Eg_.append(e)
        compton.append(Vector.from_path(file))
    if not len(compton):
        raise FileNotFoundError(f'No files found with glob pattern {pattern} in {path}')
    if Eg is not None and set(Eg_) != set(Eg):
        raise FileNotFoundError(f'Not all files found. Missing: {set(Eg) - set(Eg_)}')
    # Sort and check for equal calibration
    E, compton = zip(*sorted(zip(Eg_, compton), key=lambda x: x[0]))
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

    # Fix the lengths
    i = np.argmax([len(cmp) for cmp in compton])
    max_length = len(compton[i])
    mat = np.zeros((len(compton), max_length))
    for i, cmp in enumerate(compton):
        mat[i, :len(cmp)] = cmp.values
    return Matrix(Ex=E, Eg=compton[i].E, values=mat,
                  xlabel=r"Observed $\gamma$", ylabel=r"True $\gamma$")



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
    FE = Vector(E=E, values=df['FE'].to_numpy(), name='FE')
    SE = Vector(E=E, values=df['SE'].to_numpy(), name='SE')
    DE = Vector(E=E, values=df['DE'].to_numpy(), name='DE')
    AP = Vector(E=E, values=df['AP'].to_numpy(), name='AP')
    Eff = Vector(E=E, values=df['Eff'].to_numpy(), name='Eff')
    FWHM = None
    if read_fwhm:
        FWHM = Vector(E=E, values=df['FWHM'].to_numpy())
    return FE, SE, DE, AP, Eff, FWHM


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
    path.mkdir(parents=True, exist_ok=True)
    fname = path / 'compton.npz'
    if fname.exists() and not exists_ok:
        raise FileExistsError(f'File {fname} already exists.')
    mapping = {'values': data.compton.values, 'E_observed': data.E_observed,
               'E_true': data.E}
    np.savez(fname, **mapping)

    fields = ('FE', 'SE', 'DE', 'AP', 'Eff')
    arrays = [getattr(data, name).values for name in fields]
    E = data.E
    mapping = {k: v for k, v in zip(fields, arrays)}
    mapping['E'] = E
    fname = path / 'discrete.npz'
    if fname.exists() and not exists_ok:
        raise FileExistsError(f'File {fname} already exists.')
    np.savez(fname, **mapping)


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
