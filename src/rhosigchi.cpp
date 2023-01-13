#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include <pybind11/iostream.h>

#include<iostream>

namespace py = pybind11;


#define MAX(a, b) ( a > b ) ? a : b;
#define MIN(a, b) ( a < b ) ? a : b;

template<typename T>
class Array {
private:
    T *data;
    size_t ndim;
    size_t xsize;
    size_t ysize;
    size_t zsize;

public:
    Array(py::array_t<T, py::array::c_style | py::array::forcecast> &array)
        : data( array.mutable_data() )
        , ndim( array.ndim() )
        , xsize( ( ndim >= 1 ) ? array.shape()[0] : 1 )
        , ysize( ( ndim >= 2 ) ? array.shape()[1] : 1 )
        , zsize( ( ndim >= 3 ) ? array.shape()[2] : 1 )
        {
            if ( array.ndim() > 3 )
                throw std::runtime_error("Maximum dimentionality is 3.");
        }

    size_t Ndim() const { return ndim; }
    size_t Xsize() const {  return xsize; }
    size_t Ysize() const {  return ysize; }
    size_t Zsize() const {  return zsize; }

    T *begin(){ return data; }
    T *begin() const { return data; }

    T *end(){ return data+xsize*ysize*zsize; }
    T *end() const { return data+xsize*ysize*zsize; }


    // We will allow access to up to three axis

    T &operator()(const size_t &i){
        // Check that the array has the correct number of axis
        if ( ndim != 1 )
            throw std::runtime_error("Array have more than one axis");
        if ( i >= xsize )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__)+"', i="+std::to_string(i)+", len="+std::to_string(xsize));
        return data[i];
    }
    T &operator()(const size_t &i) const {
        // Check that the array has the correct number of axis
        if ( ndim != 1 )
            throw std::runtime_error("Array have more than one axis");
        if ( i >= xsize )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__)+"', i="+std::to_string(i)+", len="+std::to_string(xsize));
        return data[i];
    }
    T &operator()(const size_t &i, const size_t &j){
        // Check that the array has the correct number of axis
        if ( ndim != 2 )
            throw std::runtime_error("Array have more than one axis");
        if ( i >= xsize )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i * ysize + j];
    }
    T &operator()(const size_t &i, const size_t &j) const {
        // Check that the array has the correct number of axis
        if ( ndim != 2 )
            throw std::runtime_error("Array have more than one axis");
        if ( i >= xsize )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i * ysize + j];
    }
    T &operator()(const size_t &i, const size_t &j, const size_t &k){
        // Check that the array has the correct number of axis
        if ( ndim != 2 )
            throw std::runtime_error("Array have more than one axis");
        if ( i >= xsize )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i * (ysize * zsize) + j * zsize + k];
    }
    T &operator()(const size_t &i, const size_t &j, const size_t &k) const {
        // Check that the array has the correct number of axis
        if ( ndim != 2 )
            throw std::runtime_error("Array have more than one axis");
        if ( i >= xsize )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i * (ysize * zsize) + j * zsize + k];
    }


};


template<typename T>
class SimpleVector
{
private:
    std::unique_ptr<T[]> data;
    size_t xdim;
public:
    SimpleVector(const size_t &_xdim, const T *_data = nullptr)
        : data( new T[_xdim] ), xdim( _xdim )
        {
            for ( size_t i = 0 ; i < xdim ; ++i ){
                if ( _data ){
                    data[i] = _data[i];
                } else {
                    data[i] = 0;
                }
            }
        }

    T &operator()(const size_t &i){
        if ( i >= xdim )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i];
    }

    T &operator()(const size_t &i) const {
        if ( i >= xdim )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i];
    }

    T *begin(){ return data; }
    T *begin() const { return data; }
    T *end(){ return data; }
    T *end() const { return data; }

    void set_all(const T &v){
        for ( auto &val : *this ){
            val = v;
        }
    }
};

template<typename T>
class SimpleMatrix
{
private:
    std::unique_ptr<T[]> data;
    size_t xdim;
    size_t ydim;

public:
    SimpleMatrix(const size_t &_xdim, const size_t &_ydim, const T *_data = nullptr)
        : data( new T[_xdim * _ydim] )
        , xdim( _xdim ), ydim( _ydim ){
            for ( size_t i = 0 ; i < xdim ; ++i ){
                for ( size_t j = 0 ; j < ydim ; ++j ){
                    if ( _data )
                        data[i * ydim + j] = _data[i * ydim + j];
                    else
                        data[i * ydim + j] = 0;
                }
            }
        }

    T &operator()(const size_t &i, const size_t &j){
        if ( (i >= xdim) || (j >= ydim) )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i * ydim + j];
    }

    T &operator()(const size_t &i, const size_t &j) const {
        if ( (i >= xdim) || (j >= ydim) )
            throw std::runtime_error("Index out of bounds, regards from '"+std::string(__PRETTY_FUNCTION__));
        return data[i * ydim + j];
    }

    inline size_t xsize() const { return xdim; }
    inline size_t ysize() const { return ydim; }
};


void iterate(Array<double> FgN, Array<double> sFgN, Array<double> rho, Array<double> sig,
             int jmin, int jmax, int igmin, Array<int> igmax, int iu0, int nit)
{
    py::scoped_ostream_redirect stream(std::cout, py::module_::import("sys").attr("stdout"));
    // We expect that whoever calls this function knows that the input needs
    // to have the correct sizes, etc.
    double var, up, down;
    int it, ig, ix, iu, ii;
    int maxig = *std::max_element(std::begin(igmax), std::end(igmax));
    int size_use = ( maxig > (jmax+iu0) ) ? maxig : (jmax+iu0);
    SimpleVector<double> fun1(jmax), fun2(jmax), fun3(jmax);
    SimpleMatrix<double> nom(size_use, size_use), denom(size_use, size_use);
    for ( it = 1 ; it < nit ; ++it ){
        if ( it <= 5 )
            var = 1.2;
        else if ( it <= 12 )
            var = 1.1;
        else if ( it <= 21 )
            var = 1.05;
        else if ( it <= 30 )
            var = 1.025;
        else
            var = 1.01;

        for ( ix = jmin ; ix < jmax ; ++ix ){
            fun1(ix) = 0;
            fun2(ix) = 0;
            fun3(ix) = 0;
            for ( ig = igmin ; ig < igmax(ix) ; ++ig ){
                iu = ix - ig + iu0;
                fun1(ix) += sig(it-1, ig)*rho(it-1,iu);
                if ( sFgN(ix,ig) > 0 ){
                    double sr = sig(it-1, ig)*rho(it-1,iu);
                    fun2(ix) += pow(sr/sFgN(ix,ig), 2);
                    fun3(ix) += sr*FgN(ix,ig)/pow(sFgN(ix,ig), 2);
                }
            }
            if ( fun1(ix) > 0 ){
                fun2(ix) /= pow(fun1(ix), 3);
                fun3(ix) /= pow(fun1(ix), 2);
            } else {
                fun2(ix) = 0;
                fun3(ix) = 0;
            }
            for ( ig = igmin ; ig < igmax(ix) ; ++ig ){
                if ( fun1(ix)*sFgN(ix,ig) > 0 ){
                    nom(ix,ig) = fun2(ix) - fun3(ix);
                    nom(ix,ig) += FgN(ix,ig)/(fun1(ix)*pow(sFgN(ix,ig), 2));
                    denom(ix,ig) = pow(fun1(ix)*sFgN(ix,ig), -2);
                } else {
                    nom(ix,ig) = 0;
                    denom(ix,ig) = 0;
                }
            }
        }
        
        // Updating sigma
        for ( ig = igmin ; ig < maxig ; ++ig ){
            up = 0;
            down = 0;
            ii = (ig > jmin) ? ig : jmin;
            for ( ix = ii ; ix < jmax ; ++ix ){
                iu = ix - ig + iu0;
                up += rho(it-1, iu)*nom(ix, ig);
                down += rho(it-1, iu)*rho(it-1, iu)*denom(ix, ig);
            }
            if ( down > 0 ){
                if ( (up/down) > var*sig(it-1,ig) )
                    sig(it,ig) = var*sig(it-1,ig);
                else if ( (up/down) < sig(it-1,ig)/var )
                    sig(it,ig) = sig(it-1,ig)/var;
                else
                    sig(it,ig) = up/down;
            } else
                sig(it,ig) = 0;
        }
        // Updating rho
        for ( iu = 0 ; iu < jmax - igmin + iu0 ; ++iu ){
            up = 0;
            down = 0;
            ii = (iu > jmin) ? iu : jmin;
            for ( ix = ii ; ix < jmax ; ++ix ){
                ig = ix - iu + iu0;
                up += sig(it-1,ig)*nom(ix,ig);
                down += sig(it-1,ig)*sig(it-1,ig)*denom(ix,ig);
            }
            if ( down > 0 ){
                if ( (up/down) > (var*rho(it-1,iu)) )
                    rho(it,iu) = var*rho(it-1,iu);
                else if ( (up/down) < (rho(it-1,iu)/var) )
                    rho(it,iu) = rho(it-1,iu)/var;
                else
                    rho(it,iu) = up/down;
            } else
                rho(it,iu) = 0;
        }
    }
}

void iterate_proxy(py::array_t<double, py::array::c_style | py::array::forcecast> FgN,
                   py::array_t<double, py::array::c_style | py::array::forcecast> sFgN,
                   py::array_t<double, py::array::c_style | py::array::forcecast> rho,
                   py::array_t<double, py::array::c_style | py::array::forcecast> sig,
                   int jmin, int jmax, int igmin,
                   py::array_t<int, py::array::c_style | py::array::forcecast> igmax,
                   int iu0, int nit)
{
    iterate(Array(FgN), Array(sFgN), Array(rho), Array(sig),
            jmin, jmax, igmin, Array(igmax), iu0, nit);
}

double Finv(const double &y0)
{
    double xl = -3.0, xh = 3.0, x = 0, yl = 0, yh = 1.0, y = 0.5;
    while ( abs(y-y0) > 1e-3 ){
        x=xl+(xh-xl)*(y0-yl)/(yh-yl);
        y=0.5*(1.+erf(x/1.414213562));
        if ( y > y0 ){
            yl = y;
            xl = x;
        } else {
            yh = y;
            xh = x;
        }
    }
    return x;
}


PYBIND11_MODULE(rhosigchi, m){
    m.doc() = "Implementation of 'iterate' subroutine needed for the rhosigchi algorithm";
    m.def("iterate", &iterate_proxy,
        R"PREFIX(Reimplementation of the iterate subroutine from rhosigchi.
        Args:
            FgN: Normalized first generation matrix (NxN numpy array)
            sFgN: Uncertanty of normalized first generation matrix (NxN numpy array)
            rho: Array to store the NLD values (Nit x M numpy array, M is the number of finale energies)
            sig: Array to store the transmission coefficient values (Nit x M numpy array, M is the number of gamma energies)
            jmin: Lowest index on excitation axis to include in fit
            jmax: Highest index+1 on excitation axis to include in fit
            igmin: Lowest index on gamma axis to include in fit
            igmax: List of highest index+1 on gamma axis (for each excitation bin) to include in fit
            iu0: Excitation energy offset
            nit: Number of iterations
        Returns: None
        )PREFIX");
    m.def("Finv", &Finv,
        R"PREFIX(Inverting the monoton increasing function r=F(z) -> z=Finv(r).
        This means to find the z-value giving the value y0.
        The function F is the cummulative Gauss function F=1/2(1+erf(z/sqrt(2)).)PREFIX");
    m.def("Finv_vec", py::vectorize(Finv), "Vectorized version of Finv");
}
