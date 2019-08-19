from typing import Union, Iterable, List, Tuple
from pathlib import Path
import tarfile
import time
import numpy as np


def mama_read(filename):
    # Reads a MAMA matrix file and returns the matrix as a numpy array,
    # as well as a list containing the four calibration coefficients
    # (ordered as [bx, ax, by, ay] where Ei = ai*channel_i + bi)
    # and 1-D arrays of lower-bin-edge calibrated x and y values for plotting
    # and similar.
    matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
    cal = {}
    with open(filename, 'r') as datafile:
        calibration_line = datafile.readlines()[6].split(",")
        # a = [float(calibration_line[2][:-1]), float(calibration_line[3][:-1]), float(calibration_line[5][:-1]), float(calibration_line[6][:-1])]
        # JEM update 20180723: Changing to dict, including second-order term for generality:
        # print("calibration_line =", calibration_line, flush=True)
        cal = {
            "a0x": float(calibration_line[1]),
            "a1x": float(calibration_line[2]),
            "a2x": float(calibration_line[3]),
            "a0y": float(calibration_line[4]),
            "a1y": float(calibration_line[5]),
            "a2y": float(calibration_line[6])
        }
    Ny, Nx = matrix.shape
    y_array = np.linspace(0, Ny - 1, Ny)
    x_array = np.linspace(0, Nx - 1, Nx)
    # Make arrays in center-bin calibration:
    x_array = cal["a0x"] + cal["a1x"] * x_array + cal["a2x"] * x_array**2
    y_array = cal["a0y"] + cal["a1y"] * y_array + cal["a2y"] * y_array**2
    # Then correct them to lower-bin-edge:
    y_array = y_array - cal["a1y"] / 2
    x_array = x_array - cal["a1x"] / 2
    # Matrix, Eg array, Ex array
    return matrix, x_array, y_array


def mama_write(mat, filename, comment=""):
    # Calculate calibration coefficients.
    calibration = mat.calibration()
    cal = {
        "a0x": calibration['a00'],
        "a1x": calibration['a01'],
        "a2x": 0,
        "a0y": calibration['a10'],
        "a1y": calibration['a11'],
        "a2y": 0
    }
    # Convert from lower-bin-edge to centre-bin as this is what the MAMA file
    # format is supposed to have:
    cal["a0x"] += cal["a1x"] / 2
    cal["a0y"] += cal["a1y"] / 2

    # Write mandatory header:
    header_string = '!FILE=Disk \n'
    header_string += '!KIND=Spectrum \n'
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
        for i in range(4,len(lines)):
            try:
                row = np.array(lines[i].split(), dtype="double")
                resp.append(row)
            except:
                break


    resp = np.array(resp)
    # Name the columns for ease of reading
    FWHM = resp[:,1]#*6.8 # Correct with fwhm @ 1.33 MeV?
    eff = resp[:,2]
    pf = resp[:,3]
    pc = resp[:,4]
    ps = resp[:,5]
    pd = resp[:,6]
    pa = resp[:,7]

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

def load_numpy_1D(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.load(path)
    half = int(mat.shape[0]/2)
    return mat[:half], mat[half:]

def save_numpy_1D(values: np.ndarray, E: np.ndarray,
                  path: Union[str, Path]) -> None:
    mat = np.append(values, E)
    assert mat.size % 2 == 0
    np.save(path, mat)

