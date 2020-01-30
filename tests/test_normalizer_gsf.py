import pytest
import ompy as om


# derived class for easier testing of results
class NormSpin(om.NormalizerGSF):
    def spin_dist(self, Ex, J):
        return 1

@pytest.fixture
def normspin():
    return NormSpin()


# results are obtained from
# (J == 0.0) I_i = 1/2 => I_f = 1/2, 3/2
# (J == 0.5) I_i = 0  => I_f = 1  _and_  I_i = 1  => I_f = 0, 1, 2
# ...
@pytest.mark.parametrize(
        "J, result",
        [(0, 2), (0.5, 4), (1, 5), (1.5, 6), (2, 6)])
def test_spinsum(normspin, J, result):
    assert normspin.SpinSum(Ex=10, J=J) == result


@pytest.mark.parametrize(
        "J, result",
        [(1, 2), (20, 4)])
def test_spinsum2(normspin, J, result):
    assert normspin.SpinSum(Ex=10, J=J) != result
