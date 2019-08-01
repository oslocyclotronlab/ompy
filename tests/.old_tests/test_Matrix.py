import ompy
import numpy as np
import pytest
import warnings


def test_Matrix_calibration():
    # Set up est array
    E0_array = np.linspace(-200, 200, 201)
    E1_array = np.linspace(-100, 300, 101)
    counts = np.random.normal(loc=0.01*np.meshgrid(E0_array, E1_array,
                                                   indexing="ij")[0],
                              size=(len(E0_array), len(E1_array)))
    mat = ompy.Matrix(counts, E0_array, E1_array)

    # == mat.calibration() ==
    cal = mat.calibration()
    assert(cal["a00"] == mat.E0_array[0])
    assert(cal["a01"] == (mat.E0_array[1]-mat.E0_array[0]))
    assert(cal["a10"] == mat.E1_array[0])
    assert(cal["a11"] == (mat.E1_array[1]-mat.E1_array[0]))

    # Test calibration attribute check
    mat = ompy.Matrix()
    with pytest.raises(TypeError):
        mat.calibration()

    # There doesn't seem to be any non-deprecated method for testing
    # matplotlib plots.

    # # Allocate several subplots for different tests:
    # fig, (ax1, ax2) = plt.subplots(1, 2)

    # # == mat.plot() ==
    # cbar = mat.plot(ax=ax1, zscale="linear", title="plot() works")
    # fig.colorbar(cbar, ax=ax1)

    # # == mat.cut_rect() ==
    # cut_axis = 0
    # E0_limits = [E0_array[5], E0_array[-5]]
    # if E0_limits[1] <= E0_limits[0]:
    #     E0_limits[1] = E0_limits + cal["a01"]
    # mat.cut_rect(axis=cut_axis, E_limits=E0_limits, inplace=True)
    # mat.plot(ax=ax2, zscale="linear", title="cut_rect() works")


def test_Matrix_init():
        with pytest.raises(ValueError):
            matrix = np.zeros((3, 3))
            E0_array = np.zeros(2)  # Shape mismatch to matrix
            E1_array = np.zeros(3)
            mat = ompy.Matrix(matrix=matrix, E0_array=E0_array,
                              E1_array=E1_array)
        with pytest.raises(ValueError):
            matrix = np.zeros((5, 3))
            E0_array = np.zeros(5)
            E1_array = np.zeros(5)  # Shape mismatch to matrix
            mat = ompy.Matrix(matrix=matrix, E0_array=E0_array,
                              E1_array=E1_array)
        with pytest.raises(ValueError):
            matrix = np.zeros((3, 3))
            std = np.zeros((3, 2))  # Shape mismatch to matrix
            E0_array = np.zeros(3)
            E1_array = np.zeros(3)
            mat = ompy.Matrix(matrix=matrix, E0_array=E0_array,
                              E1_array=E1_array, std=std)


def test_Matrix_read_write():
    # Make a Matrix(), write it to file and read it back
    # to check that everything looks the same
    shape = (5, 3)
    matrix = np.random.uniform(low=-1, high=1, size=shape)
    E0_array = np.linspace(0, 1, shape[0])
    E1_array = np.linspace(-1, 2, shape[1])
    mat_out = ompy.Matrix(matrix=matrix, E0_array=E0_array,
                          E1_array=E1_array)
    fname = "tmp_test_readwrite.m"
    ompy.mama_write(mat_out, fname)
    mat_in = ompy.mama_read(fname)
    tol = 1e-5
    assert (np.abs(mat_out.matrix - mat_in.matrix) < tol).all()
    assert (np.abs(mat_out.E0_array - mat_in.E0_array) < tol).all()
    assert (np.abs(mat_out.E1_array - mat_in.E1_array) < tol).all()


def test_Matrix_cut_rect():
    # Sanity check should fail
    with pytest.raises(AssertionError):
        mat = ompy.Matrix().cut_rect(None, [5.001, 5])


def test_Matrix_warning():
    with pytest.warns(UserWarning):
        shape = (5, 3)
        matrix = np.random.uniform(low=-1, high=1, size=shape)
        mat = ompy.Matrix(matrix=matrix)
        mat.load("mock/Dy164_raw.m")
