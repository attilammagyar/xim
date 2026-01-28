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


def gen_tanh_param():
    """
    Let W denote the loudness where the waveshaper turns into hard-clipping.
    The antiderivative of tanh(x) is log(cosh(x)) + c (for some c).
    Hard-clipping means that ADAA will use the antiderivative of f(x) = 1 for
    positive samples and f(x) = -1 for negative ones. In order to make the
    antiderivative continuous at W, we need a value for c for which
    log(cosh(W)) + c = W. Rearranging yields c = W - log(cosh(W)).
    """

    W = effects.db_to_linear(15.0)
    c = W - np.log(np.cosh(W))
    x = np.linspace(-W - 1.0, W + 1.0, 2000)

    f = lambda x: (
        0.0
        + np.tanh(x) * ((x > -W) & (x < W))
        + 1.0 * (x >= W)
        - 1.0 * (x <= -W)
    )
    F = lambda x: (
        0.0
        + (np.log(np.cosh(x)) + c) * ((x > -W) & (x < W)) + 
        + x * (x >= W)
        - x * (x <= -W)
    )

    plt.plot(x, f(x), label="f(x)")
    plt.plot(x, F(x), label="F(x)")
    plt.plot(x[1:], (F(x[1:]) - F(x[:-1])) / (x[1:] - x[:-1]), label="dF(x)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    gen_tanh_param()
