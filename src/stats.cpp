
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

#include "erfinv.hpp"


namespace py = pybind11;
constexpr double sqrt2 = 1.41421356237309504880168872420969807856967187;


double normal_ppf(double x)
{
    return sqrt2*erfinv(2*x - 1);
}


double truncnorm_ppf(double x, double alpha = -INFINITY, double beta = INFINITY)
{
    double a = 0.5*(1 + erf(alpha/sqrt2));
    double b = 0.5*(1 + erf(beta/sqrt2));
    return sqrt2*erfinv(2*(x*(b-a) + a) - 1);
}


PYBIND11_MODULE(stats, m) {
    m.doc() = "Implementation of the inverse cumulative normal and truncated normal distribution";

    m.def("normal_ppf", &normal_ppf, R"PREFIX(Percent point function for the normal distribution.
        For a normal distribution with mean μ and width σ use following transformation:
        ```python
        def transform_normal_ppf(x, μ, σ):
            return normal_ppf(x)*σ + μ
        ```
        Args:
            x: (float) quantile.
        Returns: Inverse of the standardized cumulative normal distribution.)PREFIX");
    
    m.def("normal_ppf_vec", py::vectorize(normal_ppf), "Vectorized version of normal_ppf accepting numpy arrays.");
    

    m.def("truncnorm_ppf", &truncnorm_ppf, R"PREFIX(Percent point function for the truncated normal distribution.
        For a truncated normal distribution with mean μ, width σ, lower limit a and upper limit b, use
        following transformation:
        ```python
        def transform_truncnorm_ppf(x, μ, σ, a, b):
            α = (a - μ)/σ
            β = (b - μ)/σ
            return truncnorm_ppf(x, α, β)*σ + μ
        ```
        Args:
            x: (float) quantile.
            alpha: (float)
            beta: (float)
        Returns: Inverse of the cumulative normal distribution.)PREFIX");
    m.def("truncnorm_ppf_vec", py::vectorize(truncnorm_ppf), "Vectorized version of truncnorm_ppf accepting numpy arrays.");
}