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

import typing

import torch as th
import numpy as np


def separate(
        mix: np.ndarray,
        model: th.nn.Module,
        status_fn: callable=lambda num_done_samples: None,
        device: typing.Optional[th.device]=None,
        dtype: typing.Optional[th.dtype]=None,
        fade_seconds: typing.Optional[float]=None,
        gen_fade_fn: typing.Optional[callable]=None,
) -> tuple[np.ndarray, typing.List]:
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = th.float

    if fade_seconds is None:
        fade_seconds = min(1.0, model.input_length_seconds / 3.0)

    if gen_fade_fn is None:
        gen_fade_fn = gen_smooth_fade

    model = model.to(device=device, dtype=dtype)

    num_fade_samples = max(2, int(fade_seconds * model.sample_rate))
    fade_in, fade_out = gen_fade_fn(num_fade_samples)

    stems = np.zeros((4,) + tuple(mix.shape))
    debug_info = []
    start_idx = 0

    while start_idx < mix.shape[0]:
        status_fn(start_idx)

        end_idx = min(mix.shape[0] - 1, start_idx + model.num_input_samples)
        chunk = mix[start_idx:end_idx, :]

        if chunk.shape[0] < model.num_input_samples:
            chunk = np.pad(chunk, ((0, model.num_input_samples - chunk.shape[0]), (0, 0)))

        chunk_stems = model(
            th.from_numpy(chunk).unsqueeze(0).to(device=device, dtype=dtype)
        ).detach().squeeze(0).numpy(force=True)

        if hasattr(model, "debug_info") and model.debug_info is not None:
            debug_info.append(model.debug_info)

        if start_idx > 0:
            chunk_stems[:, :num_fade_samples, :] *= fade_in[np.newaxis, :, np.newaxis]

        end_idx = start_idx + model.num_input_samples

        if stems.shape[1] < end_idx:
            stems = np.pad(stems, ((0, 0), (0, end_idx - stems.shape[1]), (0, 0)))

        stems[:, start_idx:end_idx, :] += chunk_stems

        next_start_idx = start_idx + model.num_input_samples - num_fade_samples

        if next_start_idx < mix.shape[0]:
            stems[:, end_idx - num_fade_samples:end_idx, :] *= fade_out[np.newaxis, :, np.newaxis]

        start_idx = next_start_idx

    return stems, debug_info


def gen_linear_fade(num_fade_samples: int) -> tuple[np.ndarray]:
    x = np.linspace(0.0, 1.0, num_fade_samples)
    fade_in = x
    fade_out = 1.0 - fade_in

    return fade_in, fade_out


def gen_smooth_fade(num_fade_samples: int) -> tuple[np.ndarray]:
    x = np.linspace(0.0, 1.0, num_fade_samples)
    fade_in = ((-2.0 * x + 3.0) * x) * x
    fade_out = 1.0 - fade_in

    return fade_in, fade_out


def gen_steep_fade(num_fade_samples: int) -> tuple[np.ndarray]:
    a = -252.0;
    b = 1386.0;
    c = 3080.0;
    d = 3465.0;
    e = 1980.0;
    f = 462.0;

    x = np.linspace(0.0, 1.0, num_fade_samples)

    fade_in = (
        ((((((((((a * x + b) * x - c) * x + d) * x - e) * x + f) * x) * x) * x) * x) * x) * x
    );
    fade_out = 1.0 - fade_in

    return fade_in, fade_out
