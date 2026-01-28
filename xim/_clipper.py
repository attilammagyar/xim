# Copyright (c) 2026, Attila Magyar
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.linalg as la

from matplotlib import pyplot as plt

import effects


def gen_clipper_spline_params():
    T = effects.db_to_linear(-2.0)      # signal remains clean below -2 dBFS
    U = effects.db_to_linear(3.0)       # soft clipping up to 3 dBFS, harder clipping above
    V = effects.db_to_linear(-0.5)      # soft-clipped 3 dBFS attenuated to this level
    W = effects.db_to_linear(15.0)      # hard clipping above 15 dBFS
    a, b, c, d, e, g, h, m, n = _solve_clipper_spline(T, U, V, W)
    print(
        f"""\
Clipper parameters:

        static constexpr double T = {T};
        static constexpr double U = {U};
        static constexpr double V = {V};
        static constexpr double W = {W};
        static constexpr double a = {a};
        static constexpr double b = {b};
        static constexpr double c = {c};
        static constexpr double d = {d};
        static constexpr double e = {e};
        static constexpr double g = {g};
        static constexpr double h = {h};
        static constexpr double m = {m};
        static constexpr double n = {n};

"""
    )

    x = np.linspace(-W - 1.0, W + 1.0, 2000)
    p = lambda x: x
    q = lambda x: a * x ** 3.0 + b * x ** 2.0 + c * x + d
    r = lambda x: e * x ** 2.0 + g * x + h
    fp = lambda x: (
        0.0
        + p(x) * ((0.0 <= x) & (x < T))
        + q(x) * ((T <= x) & (x < U))
        + r(x) * (U <= x)
    )
    f = lambda x: fp(x) * (0.0 <= x) - fp(-x) * (x < 0.0)

    P = lambda x: x ** 2.0 / 2.0
    Q = lambda x: (
        a * x ** 4.0 / 4.0 + b * x ** 3.0 / 3.0 + c * x ** 2.0 / 2.0 + d * x + m
    )
    R = lambda x: e * x ** 3.0 / 3.0 + g * x ** 2.0 / 2.0 + h * x + n
    Fp = lambda x: (
        0.0
        + P(x) * ((0.0 <= x) & (x < T))
        + Q(x) * ((T <= x) & (x < U))
        + R(x) * (U <= x)
    )
    F = lambda x: Fp(x) * (0.0 <= x) + Fp(-x) * (x < 0.0)

    plt.plot(x, f(x), label="f(x)")
    plt.plot(x, F(x), label="F(x)")
    plt.plot(x[1:], (F(x[1:]) - F(x[:-1])) / (x[1:] - x[:-1]), label="dF(x)")
    plt.legend()
    plt.show()


def _solve_clipper_spline(T: float, U: float, V: float, W: float):
    """
    The wave shaper curve will use a spline between -W and W. Outside that
    region, it will hard-clip. Inside the region, the curve will not alter the
    signal between -T and T. Between T and U, and -T and -U, some distortion
    will be applied, not allowing the signal to exceed V (or -V). In the
    remaining regions, the signal will gradually reach its maximum.

    Let T, U, V, W be positive real numbers, T < U < W. Let's construct a spline
    from the following polynomials:

    p(x) = x
    q(x) = a * x^3 + b * x^2 + c * x + d
    r(x) = e * x^2 + g * x + h

    Then

    p'(x) = 1
    q'(x) = 3 * a * x^2 + 2 * b * x + c
    r'(x) = 2 * e * x + g

    Let

    f(x) = p(x) for 0 <= x < T
    f(x) = q(x) for T <= x < U
    f(x) = r(x) for U <= x
    f(x) = -f(-x) for x < 0

    Constraints:

    1. q(U) = V
    2. r(W) = 1
    3. r'(W) = 0

    To make f(x) a spline, we need:

    4. q(T) = p(T) = T
    5. q'(T) = p'(T) = 1
    6. r(U) = q(U) = V
    7. r'(U) - q'(U) = 0

    Using this spline in an ADAA wave shaper requires the antiderivatives as
    well. Let the integration constant of P(x) be 0, and let m and n be the
    integration constants for Q(x) and R(x) respectively.

    P(X) = (1/2) * x^2 + 0
    Q(X) = (1/4) * a * x^4 + (1/3) * b * x^3 + (1/2) * c * x^2 + d * x + m
    R(X) = (1/3) * e * x^3 + (1/2) * g * x^2 + h * x + n

    To make F(x) continuous:

    8. Q(T) = P(T) = (1/2) * T^2
    9. Q(U) - R(U) = 0

    Note: since f(x) is odd, F(x) must be even.

    The equations:

    1. U^3 * a + U^2 * b + U * c + d = V
    2. W^2 * e + W * g + h = 1
    3. 2 * W * e + g = 0
    4. T^3 * a + T^2 * b + T * c + d = T
    5. 3 * T^2 * a + 2 * T * b + c = 1
    6. U^2 * e + U * g + h = V
    7. 2 * e + g - 3 * U^2 * a - 2 * U * b - c = 0
    8. (1/4) * T^4 * a + (1/3) * T^3 * b + (1/2) * T^2 * c + T * d + m
         = (1/2) * T^2
    9. (1/4) * U^4 * a + (1/3) * U^3 * b + (1/2) * U^2 * c + U * d + m
         - (1/3) * U^3 * e - (1/2) * U^2 * g - U * h - n = 0
    """


    A = np.array(
        [
            #           a          b      c     d         e     g     h     m     n
            [   U * U * U,     U * U,     U,  1.0,      0.0,  0.0,  0.0,  0.0,  0.0],
            [         0.0,       0.0,   0.0,  0.0,    W * W,    W,  1.0,  0.0,  0.0],
            [         0.0,       0.0,   0.0,  0.0,  2.0 * W,  1.0,  0.0,  0.0,  0.0],
            [   T * T * T,     T * T,     T,  1.0,      0.0,  0.0,  0.0,  0.0,  0.0],
            [ 3.0 * T * T,   2.0 * T,   1.0,  0.0,      0.0,  0.0,  0.0,  0.0,  0.0],
            [         0.0,       0.0,   0.0,  0.0,    U * U,    U,  1.0,  0.0,  0.0],
            [-3.0 * U * U,  -2.0 * U,  -1.0,  0.0,      2.0,  1.0,  0.0,  0.0,  0.0],
            [
                (1.0 / 4.0) * T ** 4.0,     # a
                (1.0 / 3.0) * T ** 3.0,     # b
                (1.0 / 2.0) * T ** 2.0,     # c
                              T,            # d
                            0.0,            # e
                            0.0,            # g
                            0.0,            # h
                            1.0,            # m
                            0.0,            # n
            ],
            [
                (1.0 / 4.0) * U ** 4.0,     # a
                (1.0 / 3.0) * U ** 3.0,     # b
                (1.0 / 2.0) * U ** 2.0,     # c
                              U,            # d
               -(1.0 / 3.0) * U ** 3.0,     # e
               -(1.0 / 2.0) * U ** 2.0,     # g
                             -U,            # h
                            1.0,            # m
                           -1.0,            # n
            ],
        ]
    )
    b = np.array(
        [
                          V,
                        1.0,
                        0.0,
                          T,
                        1.0,
                          V,
                        0.0,
            (1.0 / 2.0) * T ** 2.0,
                        0.0,
        ]
    )

    return [float(x) for x in la.inv(A).dot(b)]


if __name__ == "__main__":
    gen_clipper_spline_params()
