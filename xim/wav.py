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

import wave

import numpy as np


SAMPLE_RATE = 44100
NUM_CHANNELS = 2
SAMPLE_WIDTH = 2
SAMPLE_NORM = 32767.0


def read_wav(file_name: str) -> np.ndarray:
    """
    Read samples from a 44.1 kHz, 16 bit, stereo or mono WAV file into a NumPy
    array. Mono samples are converted to stereo.
    """

    with wave.open(file_name, "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        assert sample_width == SAMPLE_WIDTH, "Only 16-bit WAV files are supported."
        assert num_channels <= NUM_CHANNELS, "Only stereo and mono WAV files are supported."
        assert sample_rate == SAMPLE_RATE, "Only 44.1 kHz sample rate is supported."

        raw_data = wf.readframes(num_frames)
        data = np.frombuffer(raw_data, dtype=np.int16)

        if num_channels == 1:
            data = np.stack([data, data], axis=1)
        else:
            data = data.reshape(-1, num_channels)

        return data.astype(np.float32) / SAMPLE_NORM


def write_wav(file_name: str, samples: np.array):
    """
    Save 44.1 kHz stereo audio samples as a 16 bit WAV file. Applies
    hard-clipping if necessary.
    """

    with wave.open(file_name, "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(
            (samples.clip(-1.0, 1.0) * SAMPLE_NORM).astype(np.int16).tobytes()
        )
