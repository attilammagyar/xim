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

# distutils: language = c++

import cython

from cython cimport boundscheck, wraparound, cdivision, floating
from libc.math cimport abs, cos, exp, log10, pow, sin
from libcpp.vector cimport vector

cimport numpy as cnp


import typing

import numpy as np


__all__ = ["apply_effects", "db_to_linear", "linear_to_db"]


cdef double PI = 3.14159265358979323846264338327950288419716939937510


def apply_effects(
        samples: np.ndarray,
        fx: typing.List[tuple[str, typing.Dict[str, typing.Any]]],
        sample_rate: int=44100,
) -> np.ndarray:
    """
    Apply the specified effects to the given stereo samples.
    - samples: shape (n, num_channels), dtype float32 or float64. (Modified in-place.)
    - fx: list of pairs of effect name strings and effect setting dicts

    Returns: the processed samples.

    Example:

    apply_effects(
        samples,
        [
            ("high_pass", {"freq": 50.0, "q": 1.0}),
            ("bell", {"freq": 5000.0, "q": 1.0, "gain_db": -3.0}),
            (
                "compressor",
                {
                    "threshold_dbfs": -3.0,
                    "ratio": 2.0,
                    "attack_seconds": 0.02,
                    "release_seconds": 0.1,
                },
            ),
            ("gain", {"gain_db": 6.0}),
            ("wave_shaper", {"curve": "tanh"}),
            ("stereo_enhancer", {"width": 1.5}),
            (
                "add_noise",
                {
                    "level_dbfs": -30.0,
                    "high_pass_freq": 50.0,
                    "low_pass_freq": 17000.0,
                },
            ),
            (
                "limiter",
                {
                    "threshold_dbfs": -3.0,
                    "limit_dbfs": -1.0,
                    "attack_seconds": 0.001,
                    "release_seconds": 0.2,
                },
            ),
            ("wave_shaper", {"curve": "clipper"}),
            ("normalize", {"target_dbfs": -1.0}),
        ],
        sample_rate=44100,
    )
    """

    fx_funcs = {
        "add_noise": add_noise,
        "bell": bell,
        "compressor": compressor,
        "gain": gain,
        "high_pass": high_pass,
        "limiter": limiter,
        "normalize": normalize,
        "stereo_enhancer": stereo_enhancer,
        "wave_shaper": wave_shaper,
    }

    for name, settings in fx:
        func = fx_funcs.get(name, None)

        if func is None:
            raise ValueError(
                f"Unknown effect: {name!r} - known ones: {', '.join(fx_funcs.keys())}"
            )

        samples = func(samples, sample_rate=sample_rate, **settings)

    return samples


@boundscheck(False)
@wraparound(False)
cpdef void check_params(cnp.ndarray samples):
    if samples.ndim != 2:
        raise ValueError(f"Samples must have shape (n, num_channels), got {samples.ndim=}")

    dtype = samples.dtype

    if dtype != np.float32 and dtype != np.float64:
        raise TypeError(f"Expected float32 or float64, got {dtype}")



cdef extern from "wscurves.cpp" namespace "":
    cdef T _db_to_linear[T](T db) noexcept nogil
    cdef T _linear_to_db[T](T linear) noexcept nogil
    cdef T _ws_curve_lookup[T](const T* const curve, const T x) noexcept nogil
    cdef T _ws_ad_curve_lookup[T](const T* const curve, const T x) noexcept nogil

    cdef const double* P_CLIPPER_AD_TABLE_D
    cdef const double* P_CLIPPER_TABLE_D
    cdef const float* P_CLIPPER_AD_TABLE_F
    cdef const float* P_CLIPPER_TABLE_F

    cdef const double* P_TANH_AD_TABLE_D
    cdef const double* P_TANH_TABLE_D
    cdef const float* P_TANH_AD_TABLE_F
    cdef const float* P_TANH_TABLE_F


@boundscheck(False)
@wraparound(False)
cpdef floating db_to_linear(floating db):
    """
    Convert dB to linear scale.
    """
    return <floating>_db_to_linear(db)


@boundscheck(False)
@wraparound(False)
cpdef floating linear_to_db(floating linear):
    """
    Convert linear scale to dB.
    """
    return _linear_to_db(linear)


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray gain(cnp.ndarray samples, floating gain_db, int sample_rate=44100):
    """
    Change the volume of audio in-place.
    """

    check_params(samples)

    samples *= <floating>_db_to_linear(gain_db)

    return samples


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray normalize(cnp.ndarray samples, floating target_dbfs, int sample_rate=44100):
    """
    Change the volume of audio in-place in order to set
    its peaking sample to the specified level.
    """

    check_params(samples)

    target_linear = <floating>_db_to_linear(target_dbfs)
    peak = np.abs(samples).max()

    if peak < 1e-6:
        return

    samples *= target_linear / peak

    return samples


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray compressor(
        cnp.ndarray samples,
        double threshold_dbfs=-3.0,
        double ratio=2.0,
        double attack_seconds=0.02,
        double release_seconds=0.1,
        int sample_rate=44100,
):
    """
    Peak based dynamic range compressor.
    - samples: shape (n, num_channels), dtype: float32 or float64. (Modified in-place.)
    - threshold_dbfs: level above which compression kicks in (dBFS).
    - ratio: how much to compress above the threshold.
    - attack_seconds: how long it takes for the compression to kick in.
    - release_seconds: how long till compression stops.
    - sample_rate: in Hz.

    Note: inter-sample peaks are not considered.

    Returns: the processed samples.
    """

    check_params(samples)

    cdef float[:, ::1] samples_f
    cdef double[:, ::1] samples_d

    dtype = samples.dtype

    if dtype == np.float32:
        samples_f = np.ascontiguousarray(samples)

        with cython.nogil:
            _compressor_proc(
                samples_f,
                <float>threshold_dbfs,
                <float>ratio,
                <float>attack_seconds,
                <float>release_seconds,
                sample_rate,
            )

        return samples

    if dtype == np.float64:
        samples_d = np.ascontiguousarray(samples)

        with cython.nogil:
            _compressor_proc(
                samples_d,
                <double>threshold_dbfs,
                <double>ratio,
                <double>attack_seconds,
                <double>release_seconds,
                sample_rate,
            )

        return samples

    raise RuntimeError("Reached unreachable code in compressor()")


@boundscheck(False)
@wraparound(False)
cdef void _compressor_proc(
        floating[:, ::1] samples,
        floating threshold_dbfs,
        floating ratio,
        floating attack_seconds,
        floating release_seconds,
        int sample_rate,
) noexcept nogil:
    cdef:
        Py_ssize_t i, c
        Py_ssize_t n = samples.shape[0]
        Py_ssize_t num_channels = samples.shape[1]
        floating threshold_linear, gain_linear, target_gain_linear
        floating channel_peak, peak, sample
        floating db_offset
        floating attack, release

    if sample_rate <= 0 or n == 0 or num_channels == 0:
        return

    attack = <floating>1.0

    if attack_seconds > 0.0:
        attack -= <floating>exp(-1.0 / (attack_seconds * sample_rate))

    release = <floating>1.0

    if release_seconds > 0.0:
        release -= <floating>exp(-1.0 / (release_seconds * sample_rate))

    threshold_linear = <floating>_db_to_linear(threshold_dbfs)
    db_offset = threshold_dbfs - threshold_dbfs / ratio
    gain_linear = 1.0
    peak = 0.0

    for i in range(n):
        channel_peak = 0.0

        for c in range(num_channels):
            sample = samples[i, c]

            if sample < 0.0:
                sample = -sample

            if sample > channel_peak:
                channel_peak = sample

        if channel_peak > peak:
            peak = channel_peak
        else:
            peak = 0.5 * (channel_peak + peak)

        target_gain_linear = (
            1.0
            if peak <= threshold_linear
            else (<floating>_db_to_linear(_linear_to_db(peak) / ratio + db_offset) / peak)
        )
        gain_linear += (
            (attack if target_gain_linear < gain_linear else release)
            * (target_gain_linear - gain_linear)
        )

        for c in range(num_channels):
            samples[i, c] *= gain_linear


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray limiter(
        cnp.ndarray samples,
        double threshold_dbfs=-3.0,
        double limit_dbfs=-1.0,
        double attack_seconds=0.001,
        double release_seconds=0.2,
        int sample_rate=44100,
):
    """
    Peak limiter.
    - samples: shape (n, num_channels), dtype: float32 or float64. (Modified in-place.)
    - threshold_dbfs: level above which compression kicks in (dBFS).
    - limit_dbfs: maximum output loudness (dBFS).
    - attack_seconds: how long it takes for the compression to kick in.
    - release_seconds: how long till compression stops.
    - sample_rate: in Hz.

    Note: inter-sample peaks are not considered. Sudden large peaks may exceed
    the specified limit.

    Returns: the processed samples.
    """

    check_params(samples)

    cdef float[:, ::1] samples_f
    cdef double[:, ::1] samples_d

    dtype = samples.dtype

    if dtype == np.float32:
        samples_f = np.ascontiguousarray(samples)

        with cython.nogil:
            _limiter_proc(
                samples_f,
                <float>threshold_dbfs,
                <float>limit_dbfs,
                <float>attack_seconds,
                <float>release_seconds,
                sample_rate,
            )

        return samples

    if dtype == np.float64:
        samples_d = np.ascontiguousarray(samples)

        with cython.nogil:
            _limiter_proc(
                samples_d,
                <double>threshold_dbfs,
                <double>limit_dbfs,
                <double>attack_seconds,
                <double>release_seconds,
                sample_rate,
            )

        return samples

    raise RuntimeError("Reached unreachable code in limiter()")


@boundscheck(False)
@wraparound(False)
cdef void _limiter_proc(
        floating[:, ::1] samples,
        floating threshold_dbfs,
        floating limit_dbfs,
        floating attack_seconds,
        floating release_seconds,
        int sample_rate,
) noexcept nogil:
    cdef:
        Py_ssize_t i, c
        Py_ssize_t n = samples.shape[0]
        Py_ssize_t num_channels = samples.shape[1]
        floating threshold_linear, limit_linear, makeup_gain_linear, gain_linear, target_gain_linear
        floating peak, sample
        floating attack, release

    if sample_rate <= 0 or n == 0 or num_channels == 0:
        return

    attack = <floating>1.0

    if attack_seconds > 0.0:
        attack -= <floating>exp(-1.0 / (attack_seconds * sample_rate))

    release = <floating>1.0

    if release_seconds > 0.0:
        release -= <floating>exp(-1.0 / (release_seconds * sample_rate))

    limit_linear = <floating>_db_to_linear(limit_dbfs)
    threshold_linear = <floating>_db_to_linear(threshold_dbfs) * limit_linear

    if threshold_linear < <floating>1e-6:
        threshold_linear = <floating>1e-6

    makeup_gain_linear = <floating>1.0 / threshold_linear
    gain_linear = makeup_gain_linear

    for i in range(n):
        peak = 0.0

        for c in range(num_channels):
            sample = samples[i, c]

            if sample < 0.0:
                sample = -sample

            if sample > peak:
                peak = sample

        target_gain_linear = (
            makeup_gain_linear if peak <= threshold_linear else (<floating>limit_linear / peak)
        )
        gain_linear += (
            (attack if target_gain_linear < gain_linear else release)
            * (target_gain_linear - gain_linear)
        )

        for c in range(num_channels):
            samples[i, c] *= gain_linear


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray wave_shaper(
        cnp.ndarray samples,
        str curve,
        int sample_rate=44100,
):
    """
    Antiderivative Anti-aliasing (ADAA) wave shaper.
    - samples: shape (n, num_channels), dtype: float32 or float64. (Modified in-place.)
    - curve: "tanh" or "clipper"
    - sample_rate: in Hz.

    Returns: the processed samples.
    """

    check_params(samples)

    cdef float[:, ::1] samples_f
    cdef double[:, ::1] samples_d

    if samples.shape[1] > 2:
        raise NotImplementedError(
            f"Too many channels, above 2 are not implemented, got {samples.shape[1]=}"
        )

    CLIPPER = "clipper"
    TANH = "tanh"
    curves = {CLIPPER, TANH}

    if curve not in curves:
        raise NotImplementedError(
            f"Curve not implemented, available curves: {sorted(curves)}, got {curve}"
        )

    dtype = samples.dtype

    if curve == CLIPPER:
        if dtype == np.float32:
            samples_f = np.ascontiguousarray(samples)

            with cython.nogil:
                _wave_shaper_proc(samples_f, P_CLIPPER_AD_TABLE_F, P_CLIPPER_TABLE_F, sample_rate)

            return samples

        if dtype == np.float64:
            samples_d = np.ascontiguousarray(samples)

            with cython.nogil:
                _wave_shaper_proc(samples_d, P_CLIPPER_AD_TABLE_D, P_CLIPPER_TABLE_D, sample_rate)

            return samples

    if curve == TANH:
        if dtype == np.float32:
            samples_f = np.ascontiguousarray(samples)

            with cython.nogil:
                _wave_shaper_proc(samples_f, P_TANH_AD_TABLE_F, P_TANH_TABLE_F, sample_rate)

            return samples

        if dtype == np.float64:
            samples_d = np.ascontiguousarray(samples)

            with cython.nogil:
                _wave_shaper_proc(samples_d, P_TANH_AD_TABLE_D, P_TANH_TABLE_D, sample_rate)

            return samples

    raise RuntimeError("Reached unreachable code in wave_shaper()")


@boundscheck(False)
@wraparound(False)
cdef void _wave_shaper_proc(
        floating[:, ::1] samples,
        const floating* const ad_curve,
        const floating* const curve,
        int sample_rate,
) noexcept nogil:
    cdef:
        Py_ssize_t i, c
        Py_ssize_t n = samples.shape[0]
        Py_ssize_t num_channels = samples.shape[1]
        floating[2] prev_sample, prev_sample_ad_curved
        floating sample, sample_ad_curved, delta

    if sample_rate <= 0 or n == 0 or num_channels == 0:
        return

    prev_sample[0] = 0.0
    prev_sample[1] = 0.0
    prev_sample_ad_curved[0] = _ws_ad_curve_lookup(ad_curve, prev_sample[0])
    prev_sample_ad_curved[1] = _ws_ad_curve_lookup(ad_curve, prev_sample[1])

    for i in range(n):
        for c in range(num_channels):
            sample = samples[i, c]
            sample_ad_curved = _ws_ad_curve_lookup(ad_curve, sample)
            delta = sample - prev_sample[c]
            prev_sample[c] = sample

            if abs(delta) < 1e06:
                prev_sample_ad_curved[c] = sample_ad_curved

                # We're supposed to calculate the average of the current and the
                # previous sample here, but since we only do this when their
                # difference is very small, we can probably get away with just
                # using one of them.

                samples[i, c] = _ws_curve_lookup(curve, sample)

                continue

            samples[i, c] = (sample_ad_curved - prev_sample_ad_curved[c]) / delta
            prev_sample_ad_curved[c] = sample_ad_curved


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray add_noise(
        cnp.ndarray samples,
        double level_dbfs=-30.0,
        double high_pass_freq=1.0,
        double low_pass_freq=22050.0,
        int sample_rate=44100,
):
    """
    Add noise.
    - samples: shape (n, num_channels), dtype: float32 or float64. (Modified in-place.)
    - level_dbfs: how much noise to add (dBFS).
    - high_pass_freq: apply a high-pass filter to the noise using this cut-off frequency.
    - low_pass_freq: apply a low-pass filter to the noise using this cut-off frequency.
    - sample_rate: in Hz.

    Returns: the processed samples.
    """

    check_params(samples)

    cdef float[:, ::1] samples_f, noise_f
    cdef double[:, ::1] samples_d, noise_d

    dtype = samples.dtype
    noise = 2.0 * np.random.rand(samples.shape[0], samples.shape[1]) - 1.0

    if dtype == np.float32:
        samples_f = np.ascontiguousarray(samples)
        noise_f = np.ascontiguousarray(noise.astype(dtype))

        with cython.nogil:
            _add_noise_proc(
                samples_f,
                noise_f,
                <float>level_dbfs,
                <float>high_pass_freq,
                <float>low_pass_freq,
                sample_rate,
            )

        return samples

    if dtype == np.float64:
        samples_d = np.ascontiguousarray(samples)
        noise_d = np.ascontiguousarray(noise.astype(dtype))

        with cython.nogil:
            _add_noise_proc(
                samples_d,
                noise_d,
                <double>level_dbfs,
                <double>high_pass_freq,
                <double>low_pass_freq,
                sample_rate,
            )

        return samples

    raise RuntimeError("Reached unreachable code in add_noise()")


@boundscheck(False)
@wraparound(False)
cdef void _add_noise_proc(
        floating[:, ::1] samples,
        floating[:, ::1] noise,
        floating level_dbfs,
        floating high_pass_freq,
        floating low_pass_freq,
        int sample_rate,
) noexcept nogil:
    # See:
    #  - https://en.wikipedia.org/wiki/Low-pass_filter#Discrete-time_realization
    #  - https://en.wikipedia.org/wiki/High-pass_filter#Discrete-time_realization
    #
    # The filters are combined using the following notation:
    #
    #  - High-pass:
    #
    #        S := sampling period length
    #        H := high-pass cut-off frequency
    #        r[n] := n-th raw sample (random noise)
    #
    #        v := 2 * pi * S * H
    #        a := 1 / (v + 1)
    #        x[n] := a * (x[n - 1] + r[n] - r[n - 1])
    #
    #  - Low-pass:
    #
    #        L := low-pass cut-off frequency
    #        t := 2 * pi * S * L
    #        w1 := t / (t + 1)
    #        w2 := 1 - w1
    #        y[n] := w1 * x[n] + (1 - w2) * y[n - 1]

    cdef:
        Py_ssize_t n = samples.shape[0]
        Py_ssize_t num_channels = samples.shape[1]

    if sample_rate <= 0 or n == 0 or num_channels == 0:
        return

    cdef:
        Py_ssize_t i, c
        floating S = 1.0 / <floating>sample_rate
        floating v = 2.0 * <floating>PI * S * high_pass_freq
        floating t = 2.0 * <floating>PI * S * low_pass_freq
        floating a = 1.0 / (v + 1.0)
        floating w1 = t / (t + 1.0)
        floating w2 = 1.0 - w1
        vector[floating] r_prev = vector[floating](num_channels, 0.0)
        vector[floating] x_prev = vector[floating](num_channels, 0.0)
        vector[floating] y_prev = vector[floating](num_channels, 0.0)
        floating r, x, y
        floating level_linear = <floating>_db_to_linear(level_dbfs)

    for i in range(n):
        for c in range(num_channels):
            r = noise[i, c]
            x = a * (x_prev[c] + r - r_prev[c])
            y = w1 * x + w2 * y_prev[c]

            samples[i, c] += level_linear * y

            r_prev[c] = r
            x_prev[c] = x
            y_prev[c] = y


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray bell(
        cnp.ndarray samples,
        double freq=300,
        double q=1.0,
        double gain_db=-3.0,
        int sample_rate=44100,
):
    """
    Bell (a.k.a. peaking) filter. Boosts or attenuates a range of frequencies.
    - samples: shape (n, num_channels), dtype: float32 or float64. (Modified in-place.)
    - freq: center frequency.
    - q: bandwidth (values close to 0.0 mean wide band, large values mean narrow band).
    - gain_db: how much to boost (positive values) or attenuate (negative values) in dB.
    - sample_rate: in Hz.

    Returns: the processed samples.
    """

    check_params(samples)

    cdef float[:, ::1] samples_f
    cdef double[:, ::1] samples_d

    dtype = samples.dtype

    if dtype == np.float32:
        samples_f = np.ascontiguousarray(samples)

        with cython.nogil:
            _bell_proc(samples_f, <float>freq, <float>q, <float>gain_db, sample_rate)

        return samples

    if dtype == np.float64:
        samples_d = np.ascontiguousarray(samples)

        with cython.nogil:
            _bell_proc(samples_d, <double>freq, <double>q, <double>gain_db, sample_rate)

        return samples

    raise RuntimeError("Reached unreachable code in bell()")


@boundscheck(False)
@wraparound(False)
cdef void _bell_proc(
        floating[:, ::1] samples,
        floating freq,
        floating q,
        floating gain_db,
        int sample_rate,
) noexcept nogil:
    # Notation:
    #    https://www.w3.org/TR/webaudio/#filters-characteristics
    #    https://www.w3.org/TR/2021/NOTE-audio-eq-cookbook-20210608/

    if abs(gain_db) < 0.01:
        return

    cdef:
        floating nyquist_freq = <floating>sample_rate / 2.0

    if freq >= nyquist_freq:
        return

    cdef:
        floating A, w0, alpha_q, alpha_q_times_A, alpha_q_over_A

    A = pow(10.0, gain_db / 40.0)
    w0 = 2.0 * <floating>PI * freq / <floating>sample_rate
    alpha_q = sin(w0) / (2.0 * q)
    alpha_q_times_A = alpha_q * A
    alpha_q_over_A = alpha_q / A

    _biquad_filter_proc(
        samples,
        1.0 + alpha_q_over_A,   # a0
        -2.0 * cos(w0),         # a1
        1.0 - alpha_q_over_A,   # a2
        1.0 + alpha_q_times_A,  # b0
        -2.0 * cos(w0),         # b1
        1.0 - alpha_q_times_A,  # b2
        sample_rate,
    )


@boundscheck(False)
@wraparound(False)
cdef void _biquad_filter_proc(
        floating[:, ::1] samples,
        floating a0,
        floating a1,
        floating a2,
        floating b0,
        floating b1,
        floating b2,
        int sample_rate,
) noexcept nogil:
    # Notation:
    #    https://www.w3.org/TR/webaudio/#filters-characteristics
    #    https://www.w3.org/TR/2021/NOTE-audio-eq-cookbook-20210608/

    cdef:
        Py_ssize_t n = samples.shape[0]
        Py_ssize_t num_channels = samples.shape[1]

    if sample_rate <= 0 or n == 0 or num_channels == 0:
        return

    cdef:
        Py_ssize_t i, c
        vector[floating] x_prev_1 = vector[floating](num_channels, 0.0)
        vector[floating] x_prev_2 = vector[floating](num_channels, 0.0)
        vector[floating] y_prev_1 = vector[floating](num_channels, 0.0)
        vector[floating] y_prev_2 = vector[floating](num_channels, 0.0)
        floating x, y

    if a0 < 1e-6:
        return

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    a0 = 1.0

    # Flipping the sign of a1 and a2 so that rendering can be
    # done with only additions and multiplications.
    a1 = -a1
    a2 = -a2

    for i in range(n):
        for c in range(num_channels):
            x = samples[i, c]
            y = b0 * x + b1 * x_prev_1[c] + b2 * x_prev_2[c] + a1 * y_prev_1[c] + a2 * y_prev_2[c]

            samples[i, c] = y

            x_prev_2[c] = x_prev_1[c]
            x_prev_1[c] = x
            y_prev_2[c] = y_prev_1[c]
            y_prev_1[c] = y


@boundscheck(False)
@wraparound(False)
cpdef cnp.ndarray high_pass(
        cnp.ndarray samples,
        double freq=300,
        double q=1.0,
        int sample_rate=44100,
):
    """
    High-pass filter. Attenuates frequencies below the cut-off frequency.
    - samples: shape (n, num_channels), dtype: float32 or float64. (Modified in-place.)
    - freq: cut-off frequency.
    - q: resonance.
    - sample_rate: in Hz.

    Returns: the processed samples.
    """

    check_params(samples)

    cdef float[:, ::1] samples_f
    cdef double[:, ::1] samples_d

    dtype = samples.dtype

    if dtype == np.float32:
        samples_f = np.ascontiguousarray(samples)

        with cython.nogil:
            _high_pass_proc(samples_f, <float>freq, <float>q, sample_rate)

        return samples

    if dtype == np.float64:
        samples_d = np.ascontiguousarray(samples)

        with cython.nogil:
            _high_pass_proc(samples_d, <double>freq, <double>q, sample_rate)

        return samples

    raise RuntimeError("Reached unreachable code in high_pass()")


@boundscheck(False)
@wraparound(False)
cdef void _high_pass_proc(
        floating[:, ::1] samples,
        floating freq,
        floating q,
        int sample_rate,
) noexcept nogil:
    # Notation:
    #    https://www.w3.org/TR/webaudio/#filters-characteristics
    #    https://www.w3.org/TR/2021/NOTE-audio-eq-cookbook-20210608/

    if freq < 0.001:
        return

    cdef:
        floating w0, alpha_q_db

    w0 = 2.0 * <floating>PI * freq / <floating>sample_rate
    alpha_q_db = sin(w0) / (2.0 * pow(10.0, q / 20.0))

    _biquad_filter_proc(
        samples,
        1.0 + alpha_q_db,       # a0
        -2.0 * cos(w0),         # a1
        1.0 - alpha_q_db,       # a2
        (1.0 + cos(w0)) / 2.0,  # b0
        -1.0 - cos(w0),         # b1
        (1.0 + cos(w0)) / 2.0,  # b2
        sample_rate,
    )


def stereo_enhancer(
        samples: np.ndarray,
        width: float,
        sample_rate: int=44100,
) -> np.ndarray:
    """
    Increase or decrease the difference between stereo channels.
    - samples: shape (n, 2), dtype: float32 or float64. (Modified in-place.)
    - width: values between 0 and 1 decrease separation, values above 1 increase it.
    - sample_rate: in Hz.

    Returns: the processed samples.
    """

    check_params(samples)

    center = (samples[:, 0] + samples[:, 1]) * 0.5
    diff = (samples[:, 0] - samples[:, 1]) * 0.5 * width

    return np.stack([center + diff, center - diff], axis=1)
