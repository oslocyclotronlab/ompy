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

#pragma once

#include <cmath>

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
