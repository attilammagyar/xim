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

import unittest

import numpy as np

from . import effects


class TestEffects(unittest.TestCase):
    @classmethod
    def apply_effects(cls, samples: np.ndarray, *args, **kwargs) -> np.ndarray:
        return effects.apply_effects(cls.fix_shape(samples), *args, **kwargs)

    @staticmethod
    def fix_shape(samples: np.ndarray) -> np.ndarray:
        if samples.ndim < 2:
            return samples.reshape((len(samples), 1))

        return samples

    def assert_np_almost_equal(self, a_samples: np.ndarray, b_samples: np.ndarray, delta: float):
        self.assertLess(
            np.max(np.abs(self.fix_shape(a_samples) - self.fix_shape(b_samples))),
            delta,
            msg=f"\n{a_samples=}\n{b_samples=}"
        )

    def assert_avg_abs_diff_small(self, a_samples: np.ndarray, b_samples: np.ndarray, delta: float):
        self.assertLess(
            np.mean(np.abs(self.fix_shape(a_samples) - self.fix_shape(b_samples))),
            delta,
            msg=f"\n{a_samples=}\n{b_samples=}"
        )

    def test_gain(self):
        samples = np.linspace(0.0, 1.0, 5)
        expected = np.linspace(0.0, 2.0, 5)
        actual = self.apply_effects(samples, [("gain", {"gain_db": 6.0})])
        self.assert_np_almost_equal(expected, actual, delta=0.01)

    def test_limiter(self):
        limiter_cfg = {
            "threshold_dbfs": -6.0,
            "limit_dbfs": 0.0,
            "attack_seconds": 0.001,
            "release_seconds": 0.2,
        }
        samples = np.array([0.0, -0.1, 0.5, -1.0, 2.0])
        expected = np.array([0.0, -0.2, 1.0, -1.0, 1.0])
        actual = self.apply_effects(samples, [("limiter", limiter_cfg)], sample_rate=10)
        self.assert_np_almost_equal(expected, actual, delta=0.01)

    def test_compressor(self):
        compressor_cfg = {
            "threshold_dbfs": -6.0,
            "ratio": 2.0,
            "attack_seconds": 0.001,
            "release_seconds": 0.2,
        }
        samples = np.array([0.0, -0.1, 0.5, -1.0, 2.0])
        expected = np.array([0.0, -0.1, 0.5, -0.7, 1.0])
        actual = self.apply_effects(samples, [("compressor", compressor_cfg)], sample_rate=10)
        self.assert_np_almost_equal(expected, actual, delta=0.01)

    def test_wave_shaper_clipper(self):
        samples = np.array([0.0, 0.1, -0.2, 0.9, -1.0, 2.0, -5.0, 9.0])
        expected = np.array([0.0, 0.1, -0.2, 0.9, -0.95, 0.95, -1.0, 1.0])
        actual = self.apply_effects(samples, [("wave_shaper", {"curve": "clipper"})])
        self.assert_np_almost_equal(expected, actual, delta=0.05)

    def test_wave_shaper_tanh(self):
        samples = np.linspace(-9.0, 9.0, 30)
        expected = np.tanh(samples)
        actual = self.apply_effects(samples, [("wave_shaper", {"curve": "tanh"})])
        self.assert_np_almost_equal(expected, actual, delta=0.001)

    def test_add_noise(self):
        noise_cfg = {
            "level_dbfs": -6.0,
            "high_pass_freq": 1.0,
            "low_pass_freq": 22050.0,
        }
        samples = np.ones(5000)
        np.random.seed(42)
        samples = self.apply_effects(samples, [("add_noise", noise_cfg)])
        self.assertAlmostEqual(1.0, samples.mean(), delta=0.01)
        self.assertGreater(samples.std(), 0.2)
        self.assertLess(samples.std(), 0.3)

    def test_normalize(self):
        samples = np.array([0.0, -0.5, 0.25, 1.0, -2.0])
        expected = np.array([0.0, -0.25, 0.125, 0.5, -1.0])
        actual = self.apply_effects(samples, [("normalize", {"target_dbfs": 0.0})])
        self.assert_np_almost_equal(expected, actual, delta=0.001)

    def test_bell(self):
        bell_cfg = {
            "freq": 500.0,
            "gain_db": 6.0,
            "q": 1.0,
        }
        sample_rate = 22050
        t_end = 0.02
        t = np.linspace(0.0, t_end, int(sample_rate * t_end + 0.5))
        sine_50_hz = np.sin(2.0 * np.pi * 50.0 * t)
        sine_500_hz = np.sin(2.0 * np.pi * 500.0 * t)
        sine_5000_hz = np.sin(2.0 * np.pi * 5000.0 * t)
        samples = 0.3 * sine_50_hz + 0.3 * sine_500_hz + 0.3 * sine_5000_hz
        expected = 0.3 * sine_50_hz + 0.6 * sine_500_hz + 0.3 * sine_5000_hz
        actual = self.apply_effects(
            samples,
            [("bell", bell_cfg)],
            sample_rate=sample_rate,
        )
        self.assert_avg_abs_diff_small(expected, actual, delta=0.03)

    def test_high_pass(self):
        high_pass_cfg = {
            "freq": 100.0,
            "q": 0.0,
        }
        sample_rate = 22050
        t_end = 0.02
        t = np.linspace(0.0, t_end, int(sample_rate * t_end + 0.5))
        sine_50_hz = np.sin(2.0 * np.pi * 50.0 * t)
        sine_500_hz = np.sin(2.0 * np.pi * 500.0 * t)
        sine_1000_hz = np.sin(2.0 * np.pi * 1000.0 * t)
        samples = 0.3 * sine_50_hz + 0.3 * sine_500_hz + 0.3 * sine_1000_hz
        expected = 0.0 * sine_50_hz + 0.3 * sine_500_hz + 0.3 * sine_1000_hz
        actual = self.apply_effects(
            samples,
            [("high_pass", high_pass_cfg)],
            sample_rate=sample_rate,
        )
        self.assert_avg_abs_diff_small(expected, actual, delta=0.07)

    def test_stereo_enhancer(self):
        c = [0.2, 0.2, 0.2]
        l = [0.05, -0.05, 0.05]
        r = [0.001, 0.001, 0.001]
        stereo_enhancer_cfg_1 = [("stereo_enhancer", {"width": 1.0})]
        stereo_enhancer_cfg_2 = [("stereo_enhancer", {"width": 2.0})]

        orig_samples = np.array(
            [
                [c[0] + l[0], c[0] + r[0]],
                [c[1] + l[1], c[1] + r[1]],
                [c[2] + l[2], c[2] + r[2]],
            ]
        )
        samples = orig_samples.copy()

        expected = np.array(
            [
                [c[0] + 1.5 * l[0] - 0.5 * r[0], c[0] + 1.5 * r[0] - 0.5 * l[0]],
                [c[1] + 1.5 * l[1] - 0.5 * r[1], c[1] + 1.5 * r[1] - 0.5 * l[1]],
                [c[2] + 1.5 * l[2] - 0.5 * r[2], c[2] + 1.5 * r[2] - 0.5 * l[2]],
            ]
        )

        actual_1 = self.apply_effects(samples, stereo_enhancer_cfg_1, sample_rate=10)
        self.assert_np_almost_equal(orig_samples, actual_1, delta=0.001)

        actual_2 = self.apply_effects(samples, stereo_enhancer_cfg_2, sample_rate=10)
        self.assert_np_almost_equal(expected, actual_2, delta=0.001)
