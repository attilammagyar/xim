#!/usr/bin/env python3

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

import os.path
import re
import sys

import numpy as np

from matplotlib import pyplot as plt



def main(argv):
    avg_re = re.compile(r"^ *train .*Avg:([-+0-9]*\.[0-9]*)")

    max_len = 0
    series = []
    labels = []

    for i, arg in enumerate(argv):
        if i == 0:
            continue

        values = []

        with open(arg, "r") as f:
            for line in f:
                if match_ := avg_re.search(line):
                    try:
                        values.append(float(match_[1]))
                    except:
                        values.append(np.nan)

            if len(values) > max_len:
                max_len = len(values)

        series.append(values)
        labels.append(os.path.basename(arg))

    for values in series:
        if len(values) < max_len:
            values.extend([np.nan] * (max_len - len(values)))

    x = np.linspace(0.0, 1.0, max_len)

    for i, values in enumerate(series):
        plt.plot(x, values, label=labels[i])

    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

