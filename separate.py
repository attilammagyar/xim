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
import sys
import typing

import numpy as np
import torch as th

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import xim
import xim.wav
import xim.model


def main(argv):
    try:
        model_file_name = argv[1]
        mix_file_name = argv[2]
        out_dir_name = argv[3]

    except Exception as error:
        print(
            f"Usage: {os.path.basename(argv[0])} model.pt mix.wav out_dir [xim|hdemucs]",
            file=sys.stderr,
        )
        print(f"{type(error)}: {error}")

        return 1

    model_name = argv[4] if len(argv) > 4 else "xim"

    model_file_base_name = os.path.basename(model_file_name)
    mix_file_base_name = os.path.basename(mix_file_name)

    with th.no_grad():
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        mix = xim.wav.read_wav(mix_file_name)

        model = xim.model.create_model(model_name, is_debugging=False).to(device)
        model.load_state_dict(th.load(model_file_name, weights_only=True, map_location=device))
        model.train(False)

        stems, debug_info = xim.separate(
            mix,
            model,
            status_fn=lambda num_done_samples: print(
                f"Separating {mix_file_base_name} with {model_file_base_name}:"
                f" {num_done_samples / mix.shape[0] * 100.0:.2f}%"
            )
        )

    base_file_name = os.path.join(
        out_dir_name,
        os.path.splitext(os.path.basename(mix_file_name))[0],
    )

    stem_names = ("drums", "bass", "vocals", "others")

    for i, stem_name in enumerate(stem_names):
        out_file_name = base_file_name + "-" + stem_name + ".wav"

        print(f"Writing {out_file_name}")

        xim.wav.write_wav(out_file_name, stems[i, :, :])

    if model_name == "xim" and len(debug_info) > 0:
        visualize_xim_debug_info(
            debug_info,
            out_dir_name,
            base_file_name,
            stem_names,
            model.sample_rate,
            model.fft_hop_length,
        )
        model_objs = tuple(find_conv2d(model.ftr_extractor.convolver, "ftr_extractor.convolver")) + (
            ("src_pos", model.src_pos),
            ("tgt_pos", model.tgt_pos),
        )
        visualize_model(model_objs, out_dir_name)

    return 0


def find_conv2d(obj, path):
    if isinstance(obj, th.nn.Conv2d):
        yield f"Conv2d ({path})", obj

    elif isinstance(obj, xim.model.SkipConnection):
        yield from find_conv2d(obj.module, f"{path}.module")

    elif isinstance(obj, th.nn.Sequential):
        for i, item in enumerate(obj):
            yield from find_conv2d(item, f"{path}[{i}]")


def visualize_xim_debug_info(
        debug_info,
        out_dir_name: str,
        base_file_name: str,
        stem_names: tuple[str],
        sample_rate: int,
        fft_hop_length: int,
):
    features = []
    tracks = []

    for chunk in debug_info:
        features.append(chunk[0].squeeze(0))
        tracks.append(chunk[1:])

    del debug_info

    features = np.array(features)
    features = features.reshape((features.shape[0] * features.shape[1], features.shape[2]))

    features_var_over_time = features.var(axis=0)
    print(f"{features_var_over_time.shape=}")
    print(f"{features_var_over_time.mean()=:.9f}")

    del features_var_over_time

    tracks = np.array(tracks)

    assert len(tracks.shape) == 6 and tracks.shape[1] == 5 and tracks.shape[2] == 1 and tracks.shape[3] == 2, \
        f"Unexpected debug info shape; expected=[N, 5, 1, 2, Freq, Time], got={tracks.shape}"

    # tracks.shape == [num_chunks, 5, batch_size, num_channels, num_freq_bins, num_time_frames]
    tracks = np.permute_dims(tracks, (2, 1, 3, 0, 5, 4))[0]

    # tracks.shape == [5, num_channels, num_chunks, num_time_frames, num_freq_bins]
    tracks = tracks.reshape(
        (
            tracks.shape[0],
            tracks.shape[1],
            tracks.shape[2] * tracks.shape[3],
            tracks.shape[4],
        ),
    )

    # tracks.shape == [5, num_channels, num_chunks * num_time_frames, num_freq_bins]
    num_tracks, num_channels, num_time_frames, num_freq_bins = tracks.shape

    track_names = tuple(f"{stem_name.title()} Mask" for stem_name in stem_names)

    dpi = 200
    fig = plt.figure(figsize=(12.0, 3.0 * 5.0), dpi=dpi)
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(
        features.T,
        cmap="grey",
        interpolation="bicubic",
        aspect="equal",
        vmin=features.min() - 1e-6,
        vmax=features.max() + 1e-6,
    )
    ax.set_title("Features")

    time = np.linspace(0.0, tracks[0].shape[1] * fft_hop_length / sample_rate, tracks[0].shape[1])
    freqs = np.geomspace(1.0, sample_rate / 2.0, tracks[0].shape[2])

    for i in range(2):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.pcolormesh(time, freqs, tracks[0, i].T.astype(np.float32), cmap="magma")
        ax.set_title(f"Mix Spectrogram, Channel {i + 1}")
        ax.set_ylabel("freq (Hz)")
        ax.set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig(base_file_name + "-mix.png", dpi=dpi, bbox_inches="tight")
    plt.clf()

    del fig, gs, ax

    time = np.linspace(0.0, tracks[0].shape[1] * fft_hop_length / sample_rate, tracks[0].shape[1] + 1)
    freqs = np.geomspace(1.0, sample_rate / 2.0, tracks[0].shape[2] + 1)

    for track_idx in range(1, num_tracks):
        track = tracks[track_idx]

        norm = track.max()

        if norm >= 1e-6:
            track /= norm

        assert track.min() >= 0.0, f"Expected track {track_idx} to be non-negative; {track.min()=}"

        red = track[0].T.astype(np.float32)
        blue = track[1].T.astype(np.float32)
        green = np.zeros_like(red)

        rgb = np.stack((red, green, blue), axis=-1)

        plt.figure(figsize=(12.0, 3.0))
        plt.pcolormesh(time, freqs, rgb)
        plt.title(track_names[track_idx - 1])
        plt.ylabel("freq (Hz)")
        plt.xlabel("time (s)")
        plt.tight_layout()
        plt.savefig(base_file_name + "-" + stem_names[track_idx - 1] + ".png", dpi=dpi, bbox_inches="tight")
        plt.clf()


def visualize_model(objs: typing.List[object], out_dir_name: str):
    cmap = "gray"

    def to_numpy_2d(t: th.Tensor) -> np.ndarray:
        arr = t.numpy(force=True)

        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {arr.shape}")

        return arr

    def tile_conv_kernels(weight: th.Tensor) -> np.ndarray:
        weights = weight.numpy(force=True)

        num_out_channels, num_in_channels, kernel_height, kernel_width = weights.shape

        grid_height = num_out_channels * (kernel_height + 2)
        grid_width = num_in_channels * (kernel_width + 2)
        grid = np.zeros((grid_height, grid_width)) + weights.max()

        for out_channel_idx in range(num_out_channels):
            for in_channel_idx in range(num_in_channels):
                y1 = out_channel_idx * (kernel_height + 2)
                y2 = y1 + kernel_height

                x1 = in_channel_idx * (kernel_width + 2)
                x2 = x1 + kernel_width

                grid[y1:y2, x1:x2] = weights[out_channel_idx, in_channel_idx]

        return grid

    panel_shapes = []
    prepared = []

    for name, obj in objs:
        if isinstance(obj, th.nn.Conv2d):
            grid = tile_conv_kernels(obj.weight)
            prepared.append(("conv2d", grid, grid.shape))
            panel_shapes.append(grid.shape)

        elif isinstance(obj, th.nn.Linear):
            weights = to_numpy_2d(obj.weight)
            prepared.append(("linear", weights, weights.shape))
            panel_shapes.append(weights.shape)

        elif isinstance(obj, th.nn.Parameter):
            params = to_numpy_2d(obj.squeeze())
            prepared.append(("params", params, params.shape))
            panel_shapes.append(params.shape)

        else:
            raise NotImplementedError(f"Unsupported object type: {type(obj)}")

    n = len(prepared)
    height = 3.5 * n
    total_width = 15.0
    fig, ax = plt.subplots(n, 1, figsize=(total_width, height), squeeze=False)

    for ax, (kind, arr, shape), (name, obj) in zip(ax, prepared, objs):
        ax = ax[0]
        vmin, vmax = float(np.min(arr)), float(np.max(arr))

        if vmin == vmax:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

        aspect = "equal"
        interpolation = "bicubic"

        if kind == "conv2d":
            interpolation = "nearest"

        ax.imshow(arr, cmap=cmap, interpolation=interpolation, aspect=aspect, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

        if kind == "conv2d":
            num_out_channels, num_in_channels, kernel_height, kernel_width = obj.weight.shape
            ax.set_title(
                f"{name} Conv2d({num_out_channels}x{num_in_channels}x{kernel_height}x{kernel_width})"
            )

        elif kind == "linear":
            out_channels, in_channels = obj.weight.shape
            ax.set_title(f"{name} Linear({out_channels}x{in_channels})")

        elif kind == "params":
            height, width = arr.squeeze().shape
            ax.set_title(f"{name} Parameter({height}x{width})")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_name, "model.png"), dpi=200, bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
