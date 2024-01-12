from typing import Union, Iterable, Any, TypeAlias, Literal
from pathlib import Path
import tarfile
import time
import numpy as np
import pandas as pd
from dataclasses import asdict
from .. import __full_version__, H5PY_AVAILABLE, Unit, ROOT_AVAILABLE, ROOT_IMPORTED
from ..version import warn_version
from .index import Index, LeftUniformIndex
from warnings import warn
from ..helpers import ensure_path
import logging
import json
from ..version import Version

from ..stubs import Pathlike, array
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)

Filetype: TypeAlias = Literal['mama', 'txt', 'tar', 'np', 'npz', 'hdf5', 'csv', 'npy', 'root']

Farray: TypeAlias = NDArray[np.float64]

@ensure_path
def mama_read(filename: Path) -> tuple[Farray, Farray] | tuple[Farray, Farray, Farray]:
    """Read 1d and 2d mama spectra/matrices

    Args:
        filename (str): Filename of matrix/spectrum

    Returns:
        2 or 3 eleement tuple containing
            - **counts** (*ndarray*): array of counts.
            - **x_array** (*ndarray*): mid-bin energies of x axis.
            - **y_array** (*ndarray, optional*): Returned only if input is 2d.
                Mid-bin energies of y-axis.

    Raises:
        ValueError: If format is wrong, ie. if the calibrations line is
            not as expected.

    """
    counts = np.genfromtxt(filename, skip_header=10, skip_footer=1,
                           encoding="latin-1")
    cal = {}
    with open(filename, 'r', encoding='latin-1') as datafile:
        calibration_line = datafile.readlines()[6].split(",")
        if len(calibration_line) == 4:
            ndim = 1
            cal = {
                "a0x": float(calibration_line[1]),
                "a1x": float(calibration_line[2]),
                "a2x": float(calibration_line[3]),
            }
        elif len(calibration_line) == 7:
            ndim = 2
            cal = {
                "a0x": float(calibration_line[1]),
                "a1x": float(calibration_line[2]),
                "a2x": float(calibration_line[3]),
                "a0y": float(calibration_line[4]),
                "a1y": float(calibration_line[5]),
                "a2y": float(calibration_line[6])
            }
        else:
            raise ValueError("File format must be wrong or not implemented.\n"
                             "Check calibration line of the Mama file")

    if ndim == 1:
        Nx = counts.shape[0]
        x_array = np.linspace(0, Nx - 1, Nx)
        # Make arrays in center-bin calibration:
        x_array = cal["a0x"] + cal["a1x"] * x_array + cal["a2x"] * x_array**2
        # counts, E array
        return counts, x_array

    elif ndim == 2:
        Ny, Nx = counts.shape
        y_array = np.linspace(0, Ny - 1, Ny)
        x_array = np.linspace(0, Nx - 1, Nx)
        # Make arrays in center-bin calibration:
        x_array = cal["a0x"] + cal["a1x"] * x_array + cal["a2x"] * x_array**2
        y_array = cal["a0y"] + cal["a1y"] * y_array + cal["a2y"] * y_array**2
        # counts, Eg array, Ex array
        return counts, x_array, y_array

    else:
        raise ValueError("File format must be wrong or not implemented.\n"
                         "Check calibration line of the Mama file")


def mama_write(mat, filename, **kwargs):
    ndim = mat.values.ndim
    if ndim == 1:
        mama_write1D(mat, filename, **kwargs)
    elif ndim == 2:
        mama_write2D(mat, filename, **kwargs)
    else:
        NotImplementedError("Mama cannot read ojects with more then 2D.")


def mama_write1D(vec, filename, _assert=True):
    # MAMA is always mid-binned ?? Inconsistent with the channel encoding??
    # vec = vec.to_mid()
    if _assert:
        assert (vec.shape[0] <= 8192),\
            "Mama cannot handle vectors with dimensions > 8192. "\
            "Rebin before saving."

    # Calculate calibration coefficients.

    # Write mandatory header:
    header_string = '!FILE=Disk \n'
    header_string += '!KIND=Spectrum \n'
    header_string += '!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n'
    header_string += '!EXPERIMENT= oslo_method_python \n'
    header_string += f'!COMMENT={vec.metadata.misc} \n'
    header_string += '!TIME=DATE:' + time.strftime("%d-%b-%y %H:%M:%S",
                                                   time.localtime()) + '   \n'
    # header_string += '!OMPYVERSION=' + __full_version__ + '   \n'
    calibration = vec._index.to_unit('keV').to_calibration()
    header_string += (
        '!CALIBRATION EkeV=6, %12.6E, %12.6E, %12.6E \n'
        % (
            calibration.a0,
            calibration.a1,
            calibration.a2
        ))
    header_string += '!PRECISION=16 \n'
    header_string += "!DIMENSION=1,0:{:4d} \n".format(vec.shape[0] - 1)
    header_string += '!CHANNEL=(0:%4d) ' % (vec.shape[0] - 1)

    footer_string = "!IDEND=\n"

    # Write matrix:
    np.savetxt(
        filename,
        vec.values,
        fmt="%-17.8E",
        delimiter=" ",
        newline="\n",
        header=header_string,
        footer=footer_string,
        comments="")


def mama_write2D(mat, filename, comment="", _assert=True):
    if _assert:
        assert (mat.shape[0] <= 2048 and mat.shape[1] <= 2048),\
            "Mama cannot handle matrixes with any of the dimensions > 2048. "\
            "Rebin before saving."
    # mat = mat.to_mid()

    # Calculate calibration coefficients.
    x_calibration = mat.X_index.to_unit('keV').to_calibration()
    y_calibration = mat.Y_index.to_unit('keV').to_calibration()

    # Write mandatory header:
    header_string = '!FILE=Disk \n'
    header_string += '!KIND=Matrix \n'
    header_string += '!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n'
    header_string += '!EXPERIMENT= oslo_method_python \n'
    header_string += '!COMMENT={:s} \n'.format(comment)
    header_string += '!TIME=DATE:' + time.strftime("%d-%b-%y %H:%M:%S",
                                                   time.localtime()) + '   \n'
    header_string += (
        '!CALIBRATION EkeV=6, %12.6E, %12.6E, %12.6E, %12.6E, %12.6E, %12.6E \n'
        % (
            y_calibration.a0,
            y_calibration.a1,
            y_calibration.a2,
            x_calibration.a0,
            x_calibration.a1,
            x_calibration.a2
        ))
    header_string += '!PRECISION=16 \n'
    header_string += "!DIMENSION=2,0:{:4d},0:{:4d} \n".format(
        mat.shape[1] - 1, mat.shape[0] - 1)
    header_string += '!CHANNEL=(0:%4d,0:%4d) ' % (mat.shape[1] - 1,
                                                  mat.shape[0] - 1)
    footer_string = "!IDEND=\n"

    # Write matrix:
    np.savetxt(
        filename,
        mat.values,
        fmt="%-17.8E",
        delimiter=" ",
        newline="\n",
        header=header_string,
        footer=footer_string,
        comments="")


def read_response(fname_resp_mat, fname_resp_dat):
    # Import response matrix
    R, cal_R, Eg_array_R, tmp = mama_read(fname_resp_mat)
    # We also need info from the resp.dat file:
    resp = []
    with open(fname_resp_dat) as file:
        # Read line by line as there is crazyness in the file format
        lines = file.readlines()
        for i in range(4, len(lines)):
            try:
                row = np.array(lines[i].split(), dtype="double")
                resp.append(row)
            except:
                break

    resp = np.array(resp)
    # Name the columns for ease of reading
    FWHM = resp[:, 1]  # *6.8 # Correct with fwhm @ 1.33 MeV?
    eff = resp[:, 2]
    pf = resp[:, 3]
    pc = resp[:, 4]
    ps = resp[:, 5]
    pd = resp[:, 6]
    pa = resp[:, 7]

    return R, FWHM, eff, pc, pf, ps, pd, pa, Eg_array_R


def save_tar(objects: Union[np.ndarray, Iterable[np.ndarray]],
             path: Union[str, Path]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tarpath = str(path) if path.suffix == '.tar' else str(path) + '.tar'

    tar = tarfile.open(tarpath, 'w')
    for num, object in enumerate(objects):
        npath = Path(str(path)[:-len(path.suffix)] + str(num) + '.npy')
        np.save(npath, object)
        tar.add(npath)
        npath.unlink()
    tar.close()


def load_tar(path: Union[str, Path]) -> list[np.ndarray]:
    if isinstance(path, str):
        path = Path(path)

    tarpath = str(path) if path.suffix == '.tar' else str(path) + '.tar'

    tar = tarfile.open(tarpath)
    objects = []
    for name in tar.getnames():
        tar.extract(name)
        objects.append(np.load(name))
        Path(name).unlink()
    return objects


def save_numpy_2D(matrix: np.ndarray, Eg: np.ndarray,
                  Ex: np.ndarray, path: Union[str, Path]):
    mat = np.empty((matrix.shape[0] + 1, matrix.shape[1] + 1))
    mat[0, 1:] = Eg
    mat[1:, 0] = Ex
    mat[1:, 1:] = matrix
    np.save(path, mat)


def load_numpy_2D(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.load(path)
    return mat[1:, 1:], mat[0, 1:], mat[1:, 0]


def save_txt_2D(matrix: np.ndarray, Eg: np.ndarray,
                Ex: np.ndarray, path: Path,
                header=None):
    if header is None:
        header = ("Format:\n"
                  " 0   Eg0    Eg1    Eg2   ...\n"
                  "Ex0  val00  val01  val02\n"
                  "Ex1  val10  val11  ...\n"
                  "Ex2  ...\n"
                  "...")
    elif header is False:
        header = None
    mat = np.empty((matrix.shape[0] + 1, matrix.shape[1] + 1))
    mat[0, 0] = -0
    mat[0, 1:] = Eg
    mat[1:, 0] = Ex
    mat[1:, 1:] = matrix
    np.savetxt(path, mat, header=header)


def load_txt_2D(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.loadtxt(path)
    return mat[1:, 1:], mat[0, 1:], mat[1:, 0]


def load_numpy_1D(path: Pathlike) -> tuple[np.ndarray, np.ndarray]:
    vec = np.load(path)
    E = vec[:, 0]
    values = vec[:, 1]
    _, col = vec.shape
    if col >= 3:
        raise ValueError(f"The file {path} contains more than 2 columns")
    return values, E


def save_numpy_1D(values: np.ndarray, E: np.ndarray,
                  path: Path) -> None:
    mat = None
    mat = np.column_stack((E, values))
    np.save(path, mat)


def save_csv_1D(values: np.ndarray, E: np.ndarray,
                path: Path) -> None:
    df = {'E': E, 'values': values}
    df = pd.DataFrame(df)
    df.to_csv(path, index=False)


def load_csv_1D(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    E = df['E'].to_numpy(copy=True)
    values = df['values'].to_numpy(copy=True)
    return values, E


def load_txt_1D(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vec = np.loadtxt(path)
    E = vec[:, 0]
    values = vec[:, 1]
    return values, E


def save_txt_1D(values: np.ndarray, E: np.ndarray,
                path: Path, header='E[keV] values') -> None:
    """ E default in keV """
    mat = None
    mat = np.column_stack((E, values))
    np.savetxt(path, mat, header=header)


def encode_dict(d: dict) -> np.ndarray:
    """ Encode a dictionary as a numpy array of bytes """
    d = transform_dict(d, lambda x: isinstance(x, Unit), str)
    d = transform_dict(d, lambda x: isinstance(x, np.ndarray), lambda x: x.tolist())
    return np.array(json.dumps(d), dtype='S')

def decode_dict(a: np.ndarray) -> dict[str, Any]:
    """ Decode a numpy array of bytes as a dictionary """
    return json.loads(a.item())

def encode_string(s: str) -> np.ndarray:
    """ Encode a string as a numpy array of bytes """
    return np.array(s, dtype='S')

def decode_string(a: np.ndarray) -> str:
    """ Decode a numpy array of bytes as a string """
    b = a.item()
    if isinstance(b, bytes):
        return b.decode()
    return b


def transform_dict(d, condition_func, transform_func):
    """
    Recursively copy a nested dictionary, applying a transformation function to the values
    if they meet a given condition.

    :param d: The nested dictionary to transform.
    :param condition_func: A function that takes a value and returns True if the condition is met.
    :param transform_func: A function that takes a value and returns its transformed value.
    :return: A new dictionary with transformed values.
    """
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively apply transformation to sub-dictionaries
            new_dict[key] = transform_dict(value, condition_func, transform_func)
        elif condition_func(value):
            # Apply the transformation function if condition is met
            new_dict[key] = transform_func(value)
        else:
            # Copy the value as is
            new_dict[key] = value
    return new_dict

def save_npz_1D(path: Path, vector, exist_ok: bool = False) -> None:
    if not exist_ok and path.exists():
        raise FileExistsError(f"File {path} already exists")
    
    index = encode_dict(vector._index.to_dict())
    meta = encode_dict(asdict(vector.metadata))
    version = encode_string(__full_version__)
    mapping = {'index': index, 'values': vector.values,
               'meta': meta, 'version': version}

    np.savez(path, **mapping)


def load_npz_1D(path: Pathlike, cls, **kwargs) -> Any:
    if use_old_version(path):
        return load_npz_1D_old(path, cls, **kwargs)

    with np.load(path, allow_pickle=False, **kwargs) as data:
        meta = decode_dict(data['meta'])
        index = Index.from_dict(decode_dict(data['index']))
        values = data['values']
        version = decode_string(data['version'])
        warn_version(version)
    return cls(X=index, values=values, **meta)


def save_npz_2D(path: Path, matrix, exist_ok: bool = False) -> None:
    mapping = {'X index': encode_dict(matrix.X_index.to_dict()),
               'Y index': encode_dict(matrix.Y_index.to_dict()),
               'values': matrix.values, 'meta': encode_dict(asdict(matrix.metadata)),
               'version': encode_string(__full_version__)}
    if not exist_ok and path.exists():
        raise FileExistsError(f"{path} already exists. Use `exist_ok=True` to overwrite")
    np.savez(path, **mapping)


def load_npz_2D(path: Pathlike, cls, **kwargs) -> Any:
    if use_old_version(path):
        return load_npz_2D_old(path, cls, **kwargs)

    with np.load(path, allow_pickle=False, **kwargs) as data:
        version = decode_string(data['version'])
        warn_version(version)
        meta = decode_dict(data['meta'])
        X_index = Index.from_dict(decode_dict(data['X index']))
        Y_index = Index.from_dict(decode_dict(data['Y index']))
        values = data['values']
    return cls(X=X_index, Y=Y_index, values=values, **meta)


def load_npz_1D_old(path: Pathlike, cls, **kwargs) -> Any:
    with np.load(path, allow_pickle=True, **kwargs) as data:
        meta = data['meta'][()]
        index = Index.from_dict(data['index'][()])
        values = data['values']
        version = data['version'].item()
        warn_version(version)
    return cls(X=index, values=values, **meta)


def load_npz_2D_old(path: Pathlike, cls, **kwargs) -> Any:
    with np.load(path, allow_pickle=True, **kwargs) as data:
        version = data['version'].item()
        warn_version(version)
        meta = data['meta'][()]
        X_index = Index.from_dict(data['X index'][()])
        Y_index = Index.from_dict(data['Y index'][()])
        values = data['values']
    return cls(X=X_index, Y=Y_index, values=values, **meta)

def use_old_version(path: Pathlike) -> bool:
    with np.load(path, allow_pickle=False) as data:
        version = data['version'].item()
        if isinstance(version, bytes):
            version = version.decode()
        version = Version.from_str(version)
        if version < Version.from_str('2.1.0'):
            # Version 2.1.0 removes pickling
            return True
        return False


def save_hdf5_2D(matrix, path: Path, exist_ok: bool = False, **kwargs) -> None:
    raise ImportError("h5py is not installed")


def load_hdf5_2D(path: Path, cls, **kwargs) -> Any:
    raise ImportError("h5py is not installed")


if H5PY_AVAILABLE:
    import h5py

    def dict_to_hdf5(h5file, dictionary: dict[str, Any], path='/'):
        for key, value in dictionary.items():
            if False:
                print('\n ======')
                print(path)
                print(key, value)
                print(h5file[path].keys())
                print(h5file[path].attrs.keys())
                print(type(value))
            match value:
                case dict():
                    h5file.create_group(path + key)
                    dict_to_hdf5(h5file, value, path + key + '/')
                case int() | float() | str() | np.number():
                    # Save simple datatypes as attributes
                    h5file[path].attrs[key] = value
                case np.ndarray() | list():
                    h5file[path + key] = value
                case Unit():
                    h5file[path].attrs[key] = str(value)
                case x:
                    raise TypeError(f"Unsupported type {type(x)}:{x}")

    def hdf5_to_dict(h5file, path='/') -> dict[str, Any]:
        dictionary = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py.Dataset):  # item is a dataset
                dictionary[key] = item[()]
            elif isinstance(item, h5py.Group):  # item is a group (nested dictionary)
                dictionary[key] = hdf5_to_dict(h5file, path + key + '/')
        for key, val in h5file[path].attrs.items():
            dictionary[key] = val
        return dictionary

    #@ensure_path
    def _save_hdf5_2D(matrix, path: Path, exist_ok: bool = False, **kwargs) -> None:
        if not exist_ok and path.exists():
            raise FileExistsError(f"{path} already exists")

        kwargs.setdefault('compression', 'gzip')

        start = time.time()
        with h5py.File(path, 'w') as f:
            f.create_group('X_index')
            dict_to_hdf5(f, matrix.X_index.to_dict(), 'X_index/')
            f.create_group('Y_index')
            dict_to_hdf5(f, matrix.Y_index.to_dict(), 'Y_index/')
            f.create_dataset('values', data=matrix.values, **kwargs)
            f.create_group('meta')
            dict_to_hdf5(f, asdict(matrix.metadata), 'meta/')
            f.attrs['version'] = __full_version__
        elapsed = time.time() - start
        LOG.debug(f"Saving {path} took {elapsed:.2f} seconds")

    #@ensure_path
    def _load_hdf5_2D(path: Path, cls, **kwargs) -> Any:
        start = time.time()
        with h5py.File(path, 'r') as f:
            version = f.attrs['version']
            warn_version(version)
            meta = hdf5_to_dict(f, 'meta/')
            X_dict = hdf5_to_dict(f, 'X_index/')
            X_index = Index.from_dict(X_dict)  # type: ignore
            Y_dict = hdf5_to_dict(f, 'Y_index/')
            Y_index = Index.from_dict(Y_dict)  # type: ignore
            values = np.array(f['values'])
        elapsed = time.time() - start
        LOG.debug(f"Loading {path} took {elapsed:.2f} seconds")
        return cls(X=X_index, Y=Y_index, values=values, **meta)

    save_hdf5_2D = _save_hdf5_2D
    load_hdf5_2D = _load_hdf5_2D

def save_root_1D(vector, path: Path, exist_ok: bool = False) -> None:
    raise ImportError("ROOT is not available")

def load_root_1D(path: Path, cls) -> Any:
    raise ImportError("ROOT is not available")

if ROOT_IMPORTED:
    import ROOT
    from ROOT import TH1D, TFile

    def save_root_1D(vector, path: Path, exist_ok: bool = False,
                     name: str | None = None, simple: bool = True) -> None:
        path = path.with_suffix('.root')
        if not exist_ok and path.exists():
            raise FileExistsError(f"{path} already exists")
        vector = vector.to_left()
        with TFile(str(path), 'recreate') as f:
            hist = TH1D('hist', 'hist', vector.shape[0], vector._index[0], vector._index[-1])
            for i, val in enumerate(vector.values):
                hist.SetBinContent(i+1, val)
            hist.GetXaxis().SetTitle(vector.xlabel)
            hist.GetYaxis().SetTitle(vector.ylabel)
            hist.SetTitle(vector.title)
            if name is not None:
                hist.SetName(name)
            hist.Write()
            
            if not simple:
                # Metadata
                d = f.mkdir('index')
                d.cd()
                index = vector._index.to_dict()
                for key, val in index.items():
                    print(key, val)
                    ROOT.TNamed(key, str(val)).Write()
                #    f.SetKey(key, val)
                d = f.mkdir('meta')
                d.cd()
                meta = asdict(vector.metadata)
                for key, val in meta.items():
                    print(key, val)
                    ROOT.TNamed(key, str(val)).Write()
                ROOT.TNamed('version', __full_version__).Write()




    def load_root_1D(path: Path, cls) -> Any:
        if not path.exists():
            if path.with_suffix('.root').exists():
                path = path.with_suffix('.root')
            # else let root throw the error
        with TFile(str(path), 'read') as f:
            hist = f.Get('hist')
            values = np.array([hist.GetBinContent(i) for i in range(1, hist.GetNbinsX()+1)])
            arr = np.linspace(hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax(), hist.GetNbinsX())
            index = LeftUniformIndex.from_array(arr)
        return cls(X=index, values=values)


Suffix: TypeAlias = Literal['.npy', '.tar', '.txt', '.csv', '.m', '.npz', '.h5', 'root']
def filetype_from_suffix(path: Path) -> Filetype | None:
    suffix = path.suffix
    match suffix:
        case '.npy':
            return 'npy'
        case '.tar':
            return 'tar'
        case '.txt':
            return 'txt'
        case '.csv':
            return 'csv'
        case '.m':
            return 'mama'
        case '.npz':
            return 'npz'
        case '.h5':
            return 'hdf5'
        case '.root':
            return 'root'
        case '':
            return ''
        case _:
            return None


def resolve_filetype(path: Path, filetype: str | None) -> tuple[Path, Filetype]:
    if filetype is None:
        filetype = filetype_from_suffix(path)
        if filetype is None:
            raise ValueError(
                f"Filetype could not be determined from suffix: {path}"
            f"Supported suffixes are {Suffix}")
        # Fallback case
        if filetype == '':
            filetype = 'npz'
            path = path.with_suffix('.npz')
    filetype = filetype.lower()
    # Numpy always adds file extension
    if filetype == 'npy' and not path.suffix:
        path = path.with_suffix('.npy')
    if filetype == 'npz' and not path.suffix:
        path = path.with_suffix('.npz')
    return path, filetype

