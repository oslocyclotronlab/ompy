from typing import Union, Iterable, List, Tuple
from pathlib import Path
import tarfile
import time
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d


def mama_read(filename: str) -> Union[Tuple[ndarray, ndarray],
                                      Tuple[ndarray, ndarray, ndarray]]:
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


def mama_write(mat, filename, comment=""):
    ndim = mat.values.ndim
    if ndim == 1:
        mama_write1D(mat, filename, comment)
    elif ndim == 2:
        mama_write2D(mat, filename, comment)
    else:
        NotImplementedError("Mama cannot read ojects with more then 2D.")


def mama_write1D(mat, filename, comment=""):
    assert(mat.shape[0] <= 8192),\
        "Mama cannot handle vectors with dimensions > 8192. "\
        "Rebin before saving."

    # Calculate calibration coefficients.
    calibration = mat.calibration()
    cal = {
        "a0x": calibration['a0'],
        "a1x": calibration['a1'],
        "a2x": 0,
    }

    # Write mandatory header:
    header_string = '!FILE=Disk \n'
    header_string += '!KIND=Spectrum \n'
    header_string += '!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n'
    header_string += '!EXPERIMENT= oslo_method_python \n'
    header_string += '!COMMENT={:s} \n'.format(comment)
    header_string += '!TIME=DATE:' + time.strftime("%d-%b-%y %H:%M:%S",
                                                   time.localtime()) + '   \n'
    header_string += (
        '!CALIBRATION EkeV=6, %12.6E, %12.6E, %12.6E \n'
        % (
            cal["a0x"],
            cal["a1x"],
            cal["a2x"],
        ))
    header_string += '!PRECISION=16 \n'
    header_string += "!DIMENSION=1,0:{:4d} \n".format(
        mat.shape[0] - 1)
    header_string += '!CHANNEL=(0:%4d) ' % (mat.shape[0] - 1)

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


def mama_write2D(mat, filename, comment=""):
    assert(mat.shape[0] <= 2048 and mat.shape[1] <= 2048),\
        "Mama cannot handle matrixes with any of the dimensions > 2048. "\
        "Rebin before saving."

    # Calculate calibration coefficients.
    calibration = mat.calibration()
    cal = {
        "a0x": calibration['a0y'],
        "a1x": calibration['a1y'],
        "a2x": 0,
        "a0y": calibration['a0x'],
        "a1y": calibration['a1x'],
        "a2y": 0
    }

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
          cal["a0x"],
          cal["a1x"],
          cal["a2x"],
          cal["a0y"],
          cal["a1y"],
          cal["a2y"],
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


def load_tar(path: Union[str, Path]) -> List[np.ndarray]:
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


def load_numpy_2D(path: Union[str, Path]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.load(path)
    return mat[1:, 1:], mat[0, 1:], mat[1:, 0]


def save_txt_2D(matrix: np.ndarray, Eg: np.ndarray,
                Ex: np.ndarray, path: Union[str, Path],
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


def load_txt_2D(path: Union[str, Path]
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.loadtxt(path)
    return mat[1:, 1:], mat[0, 1:], mat[1:, 0]


def load_numpy_1D(path: Union[str, Path]
                  ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    vec = np.load(path)
    E = vec[:, 0]
    values = vec[:, 1]
    std = None
    _, col = vec.shape
    if col >= 3:
        std = vec[:, 2]
    return values, E, std


def save_numpy_1D(values: np.ndarray, E: np.ndarray,
                  std: Union[np.ndarray, None],
                  path: Union[str, Path]) -> None:
    mat = None
    if std is None:
        mat = np.column_stack((E, values))
    else:
        mat = np.column_stack((E, values, std))
    np.save(path, mat)


def save_csv_1D(values: np.ndarray, E: np.ndarray,
                std: Union[np.ndarray, None],
                path: Union[str, Path]) -> None:
    df = {'E': E, 'values': values}
    if std is not None:
        df['std'] = std
    df = pd.DataFrame(df)
    df.to_csv(path, index=False)


def load_csv_1D(path: Union[str, Path]) -> Tuple[np.ndarray,
                                                 np.ndarray,
                                                 Union[np.ndarray, None]]:
    df = pd.read_csv(path)
    E = df['E'].to_numpy(copy=True)
    values = df['values'].to_numpy(copy=True)
    std = None
    if 'std' in df.columns:
        std = df['std'].to_numpy(copy=True)
    return values, E, std


def load_txt_1D(path: Union[str, Path]
                ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    vec = np.loadtxt(path)
    E = vec[:, 0]
    values = vec[:, 1]
    std = None
    _, col = vec.shape
    if col >= 3:
        std = vec[:, 2]
    return values, E, std


def save_txt_1D(values: np.ndarray, E: np.ndarray,
                std: Union[np.ndarray, None],
                path: Union[str, Path], header='E[keV] values') -> None:
    """ E default in keV """
    mat = None
    if std is None:
        mat = np.column_stack((E, values))
    else:
        mat = np.column_stack((E, values, std))
    np.savetxt(path, mat, header=header)


def filetype_from_suffix(path: Path) -> str:
    suffix = path.suffix
    if suffix == '.tar':
        return 'tar'
    elif suffix == '.npy':
        return 'numpy'
    elif suffix == '.txt':
        return 'txt'
    elif suffix == '.m':
        return 'mama'
    elif suffix == '.csv':
        return 'csv'
    else:
        return "unknown"


def load_discrete(path: Union[str, Path], energy: ndarray,
                  resolution: float = 0.1) -> Tuple[ndarray, ndarray]:
    """Load discrete levels and apply smoothing

    Assumes linear equdistant binning

    Args:
        path (Union[str, Path]): The file to load
        energy (ndarray): The binning to use
        resolution (float, optional): The resolution (FWHM) to apply to the
            gaussian smoothing. Defaults to 0.1.

    Returns:
        Tuple[ndarray, ndarray]
    """
    energies = np.loadtxt(path)
    energies /= 1e3  # convert to MeV
    if len(energies) > 1:
        assert energies.mean() < 20, "Probably energies are not in keV"

    binsize = energy[1] - energy[0]
    bin_edges = np.append(energy, energy[-1] + binsize)
    bin_edges -= binsize / 2

    hist, _ = np.histogram(energies, bins=bin_edges)
    hist = hist.astype(float) / binsize  # convert to levels/MeV

    if resolution > 0:
        resolution /= 2.3548
        smoothed = gaussian_filter1d(hist, sigma=resolution / binsize)
    else:
        smoothed = None
    return hist, smoothed
