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

import gc
import os.path
import sys
import typing

import numpy as np
import torch as th

import xim.data
import xim.model


def main(argv):
    try:
        model_file_name = argv[1]
        test_dir_name = argv[2]

    except Exception as error:
        print(
            f"Usage: {os.path.basename(argv[0])} model.pt test_dir [xim|hdemucs]",
            file=sys.stderr,
        )
        print(f"{type(error)}: {error}")

        return 1

    model_name = argv[3] if len(argv) > 3 else "xim"

    songs = xim.data.collect_songs_stems(test_dir_name)

    with th.no_grad():
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        model = xim.model.create_model(model_name).to(device)
        model.load_state_dict(th.load(model_file_name, weights_only=True, map_location=device))
        model.train(False)

        print(
            "Song"
            "\tDrums L1\tBass L1\tVocals L1\tOthers L1"
            "\tDrums SDR\tBass SDR\tVocals SDR\tOthers SDR"
            "\tAll L1\tAll SDR"
        )

        num_channels = 2

        for i, (song_name, song_info) in enumerate(songs.items()):
            gc.collect()

            song_stems, num_samples = xim.data.load_song(song_info)

            drums = build_stem(song_stems.drums, num_samples, num_channels)
            bass = build_stem(song_stems.bass, num_samples, num_channels)
            vocals = build_stem(song_stems.vocals, num_samples, num_channels)
            others = build_stem(song_stems.others, num_samples, num_channels)

            mix = (
                build_stem(song_stems.mix, num_samples, num_channels)
                if len(song_stems.mix) > 0
                else (drums + bass + vocals + others)
            )

            del song_stems

            pred, debug_info = xim.separate(mix, model)

            del mix

            target = np.stack((drums, bass, vocals, others))

            del drums, bass, vocals, others

            l1, sdr = compute_metrics(pred, target)

            print(
                f"{song_name}"
                f"\t{l1[0]:.3f}\t{l1[1]:.3f}\t{l1[2]:.3f}\t{l1[3]:.3f}"
                f"\t{sdr[0]:.3f}\t{sdr[1]:.3f}\t{sdr[2]:.3f}\t{sdr[3]:.3f}"
                f"\t{l1.mean():.3f}\t{sdr.mean():.3f}"
            )

    return 0


def build_stem(stems: typing.List[np.ndarray], num_samples: int, num_channels) -> np.ndarray:
    samples = np.zeros((num_samples, num_channels))

    for stem in stems:
        samples += stem

    return samples


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    # SDR from AIcrowd/music-demixing-challenge-starter-kit (license: MIT)

    assert len(pred.shape) == 3 and pred.shape[0] == 4 and pred.shape[2] == 2, \
        f"Expected pred shape: [drums|bass|vocals|others, N, 2], got {pred.shape}"

    assert len(target.shape) == 3 and target.shape[0] == 4 and target.shape[2] == 2, \
        f"Expected target shape: [drums|bass|vocals|others, N, 2], got {target.shape}"

    if target.shape[1] < pred.shape[1]:
        target = np.pad(target, ((0, 0), (0, pred.shape[1] - target.shape[1]), (0, 0)))

    if pred.shape[1] < target.shape[1]:
        pred = np.pad(pred, ((0, 0), (0, target.shape[1] - pred.shape[1]), (0, 0)))

    diff = target - pred
    diff_sqr = np.square(diff)

    l1 = np.mean(np.abs(diff), axis=(1, 2))

    eps = 1e-6
    signal = eps + np.sum(np.square(target), axis=(1, 2))
    distortion = eps + np.sum(np.square(diff), axis=(1, 2))
    sdr = 10.0 * np.log10(signal / distortion)

    return l1, sdr


if __name__ == "__main__":
    sys.exit(main(sys.argv))
