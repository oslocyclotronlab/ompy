
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;

double erfinv(double x);
double normal(double x);
double truncnorm(double x, double alpha, double beta);

constexpr double sqrt2 = 1.41421356237309504880168872420969807856967187;
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


double normal(double x)
{
    return sqrt2*erfinv(2*x - 1);
}


double truncnorm(double x, double alpha = -INFINITY, double beta = INFINITY)
{
    double a = 0.5*(1 + erf(alpha/sqrt2));
    double b = 0.5*(1 + erf(beta/sqrt2));
    return sqrt2*erfinv(2*(x*(b-a) + a) - 1);
}



double erfinv(double x)
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


PYBIND11_MODULE(stats, m) {
    m.doc() = "Implementation of quantile functions in C++";

    m.def("normal", &normal, "Transforms a uniform RV [0,1] number to a Gaussian RV.");
    m.def("normal_vec", py::vectorize(normal), "Vectorized version of normal.");
    m.def("truncnorm", &truncnorm, "Transforms a uniform RV [0,1] number to a truncateed Gaussian RV.");
    m.def("truncnorm_vec", py::vectorize(truncnorm), "Vectorized version of truncnorm.");
}