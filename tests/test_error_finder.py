import pytest
import ompy as om
import numpy as np
import pymc3 as pm
from scipy.interpolate import interp1d
from numpy.testing import assert_equal, assert_allclose

def generate_data(resp_path):

    Ex_min = 4400 # keV
    Ex_max = 7700 # keV
    Eg_min = 1300 # keV
    Eg_max = Ex_max + 200 # keV
    ensemble_size = 10
    regenerate = False

    raw = om.example_raw('Dy164')
    raw.cut_diagonal(E1=(800, 0), E2=(7500, 7300))
    raw.cut('Ex', 0, 8400)

    try:
        trapezoid_cut = om.Action('matrix')
        trapezoid_cut.trapezoid(Ex_min=Ex_min, Ex_max=Ex_max, Eg_min=Eg_min, Eg_max=Eg_max, inplace=True)
        ensemble = om.Ensemble(raw=raw)
        ensemble.unfolder = om.Unfolder(response=raw)
        ensemble.first_generation_method = om.FirstGeneration()
        ensemble.generate(ensemble_size, regenerate=regenerate)
        extractor = om.Extractor(ensemble=ensemble)
        extractor.trapezoid = trapezoid_cut
        extractor.extract_from(regenerate=regenerate)
        return extractor.nld, extractor.gsf
    except:
        pass

    Eg = raw.Eg
    fwhm_abs = 90.44  # (90/1330 = 6.8%)
    response = om.Response(resp_path)
    R_ompy_unf, R_tab_unf = response.interpolate(Eg, fwhm_abs=fwhm_abs/10, return_table=True)

    fthreshold = interp1d([30., 80., 122., 183., 244., 294., 344., 562., 779., 1000.],
                          [0.0, 0.0, 0.0, 0.06, 0.44, 0.60, 0.87, 0.99, 1.00, 1.00],
                          fill_value="extrapolate")

    def apply_detector_threshold(response, table, fthreshold):
        thres = fthreshold(response.Eg)
        response.values = response.values * thres
        # renormalize
        response.values = om.div0(response.values, response.values.sum(axis=1)[:, np.newaxis])
        table["eff_tot"] *= thres
    apply_detector_threshold(R_ompy_unf, R_tab_unf, fthreshold)

    unfolder = om.Unfolder(response=R_ompy_unf)
    firstgen = om.FirstGeneration()
    unfolder.use_compton_subtraction = True  # default
    unfolder.response_tab = R_tab_unf

    unfolder.FWHM_tweak_multiplier = {"fe": 1., "se": 1.1,
                                      "de": 1.3, "511": 0.9}

    trapezoid_cut = om.Action('matrix')
    trapezoid_cut.trapezoid(Ex_min=Ex_min, Ex_max=Ex_max, Eg_min=Eg_min, Eg_max=Eg_max, inplace=True)
    E_rebinned = np.arange(100., 8500, 200)
    ensemble = om.Ensemble(raw=raw)

    ensemble.unfolder = unfolder
    ensemble.first_generation_method = firstgen
    ensemble.generate(ensemble_size, regenerate=regenerate)
    ensemble.rebin(E_rebinned, member="firstgen")

    extractor = om.Extractor(ensemble=ensemble)
    extractor.trapezoid = trapezoid_cut
    extractor.suppress_warning = True
    extractor.extract_from(regenerate=regenerate)

    return extractor.nld, extractor.gsf


def keep_only(self, vecs):
    """ Takes a list of vectors and returns a list of vectors
    where only the points shared between all vectors are returned.
    """

    E = [vec.E.copy() for vec in vecs]
    energies = {}
    for vec in vecs:
        for E in vec.E:
            if E not in energies:
                energies[E] = [False] * len(vecs)

    # Next we will add if the point is present or not
    for n, vec in enumerate(vecs):
        for E in vec.E:
            energies[E][n] = True

    keep_energy = []
    for key in energies:
        if np.all(energies[key]):
            keep_energy.append(key)

    vec_all_common = [vec.copy() for vec in vecs]
    for vec in vec_all_common:
        E = []
        values = []
        for e, value in zip(vec.E, vec.values):
            if e in keep_energy:
                E.append(e)
                values.append(value)
        vec.E = np.array(E)
        vec.values = np.array(values)

    return vec_all_common


def condition_data(_nlds, _gsfs):
    """ Ensures that data are copied and that all values are in correct
    units. It also checks that all lengths are correct.
    Args:
        nlds (List[Vector]): List of the NLDs in an ensemble
        gsfs (List[Vector]): List of the γSFs in an ensemble
    Returns:
        Tuple of nld energy points, gsf energy points and the observed
        matrix for the NLD and γSF.
    Raises:
        IndexError: If the length of members of the ensemble have an
            equal number of points in the NLD and γSF.
    Warnings:
        Will raise a warning if there are members in the ensemble that
        contains one or more nan's. This is mostly to inform the user
        and shouldn't be an issue later on.
    TODO:
        - Mitigation when cut_nan() results in different length vectors
          of different members of the ensemble. (e.g. len(nld[0]) = 10,
          len(nld[1]) = 9).
    """

    # Ensure that the same number of NLDs and GSFs are provided
    assert len(_nlds) == len(_gsfs), \
        "Number of nlds and gsfs is different" 

    N = len(_nlds)

    print(f"Processing an ensemble with {N} members")

    # Make copy to ensure that we don't overwrite anything
    nlds = [nld.copy() for nld in _nlds]
    gsfs = [gsf.copy() for gsf in _gsfs]

    print(f"Before removing nan: {len(nlds[0])} NLD values and "
          f"{len(gsfs[0])} GSF values")

    # Ensure that we have in MeV and that there isn't any nan's.
    is_nan = False
    for nld, gsf in zip(nlds, gsfs):
        nld.to_MeV()
        gsf.to_MeV()
        is_nan = np.isnan(nld.values).any() or np.isnan(gsf.values).any()
        nld.cut_nan()
        gsf.cut_nan()
    if is_nan:
        print(f"NLDs and/or γSFs contains nan's."
              " They will be removed")

    # Next we will ensure that all members have all energies
    if (not om.error_finder.all_equal([len(nld) for nld in nlds]) or
       not om.error_finder.all_equal([len(gsf) for gsf in gsfs])):
        print("Some members of the ensemble have different lengths. "
              "Consider re-binning or changing limits.")
        nlds = keep_only(nlds)
        gsfs = keep_only(gsfs)

    print(f"After removing nan: {len(nlds[0])} NLD values and "
          f"{len(gsfs[0])} GSF values")

    # Next we can extract the important parts
    E_nld = nlds[0].E.copy()
    E_gsf = gsfs[0].E.copy()

    M_nld = len(E_nld)
    M_gsf = len(E_gsf)

    nld_obs = []
    gsf_obs = []

    idx_nld = np.tile(np.arange(M_nld, dtype=int), (N-1, 1))
    idx_gsf = np.tile(np.arange(M_gsf, dtype=int), (N-1, 1))

    for n, (nld, gsf) in enumerate(zip(nlds, gsfs)):

        # Make a copy of the values arrays
        nld_array = [nld.values.copy() for nld in nlds]
        gsf_array = [gsf.values.copy() for gsf in gsfs]

        del nld_array[n]
        del gsf_array[n]

        nld_array = np.array(nld_array)
        gsf_array = np.array(gsf_array)

        # Set the observed data
        nld_obs.append(nld.values[idx_nld]/nld_array)
        gsf_obs.append(gsf.values[idx_gsf]/gsf_array)

    nld_obs = np.array(nld_obs)
    gsf_obs = np.array(gsf_obs)

    idx_coef_nld = np.repeat(np.arange(N-1), M_nld).reshape(N-1, M_nld)
    idx_coef_gsf = np.repeat(np.arange(N-1), M_gsf).reshape(N-1, M_gsf)

    idx_vals_nld = np.array([idx_nld] * N)
    idx_vals_gsf = np.array([idx_gsf] * N)

    return E_nld, E_gsf, nld_obs, gsf_obs, \
        idx_coef_nld, idx_coef_gsf, idx_vals_nld, idx_vals_gsf


def test_error_finder():
    nlds, gsfs = None, None
    try:
        nlds, gsfs = generate_data("../OCL_response_functions/nai2012_for_opt13")
    except:
        nlds, gsfs = generate_data("OCL_response_functions/nai2012_for_opt13")

    E_nld, E_gsf, nld_obs, gsf_obs, \
        idx_coef_nld, idx_coef_gsf, idx_vals_nld, idx_vals_gsf = condition_data(nlds, gsfs)

    N = len(nlds)
    e_nld, q_nld = om.error_finder.format_data(nlds)
    e_gsf, q_gsf = om.error_finder.format_data(gsfs)

    M_nld = len(e_nld)
    M_gsf = len(e_gsf)

    c_mask_nld, v_mask_nld = om.error_finder.format_mask(N, M_nld)
    c_mask_gsf, v_mask_gsf = om.error_finder.format_mask(N, M_gsf)

    assert_equal(e_nld, E_nld)
    assert_equal(e_gsf, E_gsf)

    assert_equal(q_nld, nld_obs)
    assert_equal(q_gsf, gsf_obs)

    assert_equal(c_mask_nld, idx_coef_nld)
    assert_equal(v_mask_nld, idx_vals_nld)

    assert_equal(c_mask_gsf, idx_coef_gsf)
    assert_equal(v_mask_gsf, idx_vals_gsf)
