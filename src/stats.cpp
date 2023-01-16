
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

/******** Included 'external' code ********/
/*
   libit - Library for basic source and channel coding functions
   Copyright (C) 2005-2005 Vivien Chappelier, Herve Jegou

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public
   License as published by the Free Software Foundation; either
   version 3 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public
   License along with this library; if not, write to the Free
   Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
/*
    Implementation of inverse error function taken from libit on
    3 nov. 2020. Redistributed with GPLv3.
 */


constexpr double erfinv_a3 = -0.140543331;
constexpr double erfinv_a2 = 0.914624893;
constexpr double erfinv_a1 = -1.645349621;
constexpr double erfinv_a0 = 0.886226899;
 
constexpr double erfinv_b4 = 0.012229801;
constexpr double erfinv_b3 = -0.329097515;
constexpr double erfinv_b2 = 1.442710462;
constexpr double erfinv_b1 = -2.118377725;
constexpr double erfinv_b0 = 1;

constexpr double erfinv_c3 = 1.641345311;
constexpr double erfinv_c2 = 3.429567803;
constexpr double erfinv_c1 = -1.62490649;
constexpr double erfinv_c0 = -1.970840454;

constexpr double erfinv_d2 = 1.637067800;
constexpr double erfinv_d1 = 3.543889200;
constexpr double erfinv_d0 = 1;

/* Compute the inverse error function */
inline double erfinv(double x)
{

    if ( x <= -1 ){
        return -INFINITY;
    } else if ( x >= 1 ) {
        return INFINITY;
    }

    double x2, r, y;
    double sign_x = ( x < 0 ) ? -1 : 1;
    x = fabs(x);

    if ( x <= 0.7 ){

        x2 = x * x;
        r = x * (((erfinv_a3 * x2 + erfinv_a2) * x2 + erfinv_a1) * x2 + erfinv_a0);
        r /= (((erfinv_b4 * x2 + erfinv_b3) * x2 + erfinv_b2) * x2 +erfinv_b1) * x2 + erfinv_b0;
    } else {
        y = sqrt(-log((1-x)/2.0));
        r = (((erfinv_c3 * y + erfinv_c2) * y + erfinv_c1) * y + erfinv_c0);
        r /= ((erfinv_d2 * y + erfinv_d1) * y + erfinv_d0);
    }

    r = r * sign_x;
    x = x * sign_x;

    r -= (erf (r) - x) / (2 / sqrt (M_PI) * exp (-r * r));
    r -= (erf (r) - x) / (2 / sqrt (M_PI) * exp (-r * r));

    return r;
}
/******** End 'external' code ********/



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