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

import torch as th
import torchaudio as tha


def create_model(model_name: str, is_debugging: bool=False) -> th.nn.Module:
    if model_name == "xim":
        return Xim(is_debugging=is_debugging)

    if model_name == "hdemucs":
        return HDemucs()

    raise NotImplementedError(f"Model not available; {model_name=!r}")


class Xim(th.nn.Module):
    def __init__(
            self,
            num_features: int=384,
            input_length_seconds: float=5.0,
            sample_rate: int=44100,
            num_channels: int=2,
            n_fft: int=2048,
            fft_hop_length: int=768,
            fft_window_length: int=2048,
            is_debugging=False,
    ):
        super().__init__()

        self.debug_info = None
        self.is_debugging = bool(is_debugging)

        self.num_features = int(num_features)

        self.input_length_seconds, self.sample_rate, self.num_channels = check_params(
            input_length_seconds,
            sample_rate,
            num_channels,
        )

        self.n_fft = int(n_fft)
        self.fft_hop_length = int(fft_hop_length)
        self.fft_window_length = int(fft_window_length)
        self.fft_window = th.hann_window(self.fft_window_length)

        self.num_input_samples = seconds_to_num_samples(
            self.input_length_seconds,
            self.sample_rate,
            self.fft_hop_length,
        )

        # Formulas from the docs of torch.stft()
        self.num_time_frames = int(self.num_input_samples // self.fft_hop_length + 1)
        self.num_freq_bins = int(self.n_fft // 2 + 1)

        self.ftr_extractor = FeatureExtractor(
            num_channels=self.num_channels,
            num_freq_bins=self.num_freq_bins,
            num_time_frames=self.num_time_frames,
            num_features=self.num_features,
        )

        self.pos = th.nn.Parameter(th.randn(1, self.num_time_frames, self.num_features))

        self.ftr_encoder = FeatureEncoder(
            num_time_frames=self.num_time_frames,
            num_features=self.num_features,
        )

        self.time_encoder = TimeEncoder(
            num_time_frames=self.num_time_frames,
            num_features=self.num_features,
        )

        self.stem_extractors = th.nn.ModuleList(
            [StemExtractor(num_features=self.num_features) for i in range(3)]
        )

        self.mask_decoder = MaskDecoder(
            num_features=self.num_features,
            num_freq_bins=self.num_freq_bins,
        )

    def forward(self, audio):
        # audio.shape == [batch_size, num_samples, num_channels]

        # mag.shape == phase.shape == [batch_size, num_channels, num_freq_bins, num_time_frames]
        mag, phase = self.audio_to_spectrogram(audio)

        # ftrs.shape == [batch_size, num_time_frames, num_features]
        ftrs = self.ftr_extractor(mag, phase)
        ftrs_with_pos = ftrs + self.pos

        if self.is_debugging:
            self.debug_info = [ftrs.numpy(force=True), mag.numpy(force=True)]

        memory = ftrs_with_pos + self.ftr_encoder(ftrs_with_pos) + self.time_encoder(ftrs_with_pos)
        masks = []

        for stem_extractor in self.stem_extractors:
            delta = stem_extractor(ftrs_with_pos, memory)
            ftrs_with_pos = ftrs_with_pos - delta
            ftrs = ftrs - delta
            masks.append(self.mask_decoder(delta))

        masks.append(self.mask_decoder(ftrs))

        if self.is_debugging:
            self.debug_info += [mask.numpy(force=True) for mask in masks]

        stems_mag = th.stack([mag * mask for mask in masks], dim=1)
        stems_phase = th.stack([phase for i in range(stems_mag.shape[1])], dim=1)

        return self.spectra_to_audio(stems_mag, stems_phase)

    def audio_to_spectrogram(self, audio: th.Tensor) -> th.Tensor:
        batch_size, num_samples, num_channels = audio.shape

        self.fft_window = self.fft_window.to(device=audio.device, dtype=audio.dtype)

        transformed = th.stft(
            audio.permute(0, 2, 1).reshape(batch_size * num_channels, num_samples),
            n_fft=self.n_fft,
            hop_length=self.fft_hop_length,
            win_length=self.fft_window_length,
            window=self.fft_window,
            return_complex=True,
            normalized=True,
            onesided=True,
        ).reshape(batch_size, num_channels, self.num_freq_bins, self.num_time_frames)

        mag = th.log1p(th.abs(transformed))
        phase = th.angle(transformed)

        return mag, phase

    def spectra_to_audio(self, stems_mag, stems_phase):
        batch_size, num_stems, num_channels, num_freq_bins, num_time_frames = stems_mag.shape

        self.fft_window = self.fft_window.to(device=stems_mag.device, dtype=stems_mag.dtype)

        stems_mag = th.exp(stems_mag) - 1.0
        stems_complex = th.polar(stems_mag, stems_phase)

        stems = th.istft(
            stems_complex.reshape(
                batch_size * num_stems * num_channels,
                num_freq_bins,
                num_time_frames
            ),
            n_fft=self.n_fft,
            hop_length=self.fft_hop_length,
            win_length=self.fft_window_length,
            window=self.fft_window,
            normalized=True,
            onesided=True,
        )

        num_samples = stems.shape[-1]
        stems = stems.reshape(batch_size, num_stems, num_channels, num_samples).permute(0, 1, 3, 2)

        return stems  # [batch_size, num_stems, num_samples, num_channels]


def check_params(input_length_seconds: float, sample_rate: int, num_channels: int):
    input_length_seconds = float(input_length_seconds)
    sample_rate = int(sample_rate)
    num_channels = int(num_channels)

    assert input_length_seconds > 0.0, f"input_length_seconds must be a positive float, got {input_length_seconds=!r}"
    assert sample_rate > 0, f"sample_rate must be a positive integer, got {sample_rate=!r}"

    return input_length_seconds, sample_rate, num_channels


def seconds_to_num_samples(seconds: float, sample_rate: int, fft_hop_length: int) -> int:
    num_samples = int(seconds * sample_rate) + 1
    num_samples += fft_hop_length - num_samples % fft_hop_length

    return num_samples


def init_weights(module):
    if isinstance(module, th.nn.Linear):
        th.nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:
            th.nn.init.constant_(module.bias, 0.0)


class FeatureExtractor(th.nn.Module):
    def __init__(
            self,
            num_channels: int,
            num_freq_bins: int,
            num_time_frames: int,
            num_features: int,
    ):
        super().__init__()

        self.num_channels = int(num_channels)
        self.num_freq_bins = int(num_freq_bins)
        self.num_time_frames = int(num_time_frames)
        self.num_features = int(num_features)

        conv_layers, num_conv_channels, num_conv_bins = self._build_conv_layers()
        self.convolver = th.nn.Sequential(*conv_layers)
        self.conv_out_dim = num_conv_channels * num_conv_bins

        self.project = th.nn.Linear(
            in_features=self.conv_out_dim,
            out_features=self.num_features
        )

    def _build_conv_layers(self):
        num_conv_ch = 3 * self.num_channels

        num_conv_ch_next = 6 * self.num_channels
        num_conv_bins = self.num_freq_bins

        conv_layers = []

        # TODO: check if normalization would be useful.

        for i in range(3):
            conv_layers += [
                th.nn.Conv2d(in_channels=num_conv_ch, out_channels=num_conv_ch_next, kernel_size=1),
                th.nn.GELU(),
                SkipConnection(
                    th.nn.Sequential(
                        th.nn.Conv2d(in_channels=num_conv_ch_next, out_channels=num_conv_ch_next, kernel_size=3, padding=1),
                        th.nn.GELU(),
                        th.nn.Conv2d(in_channels=num_conv_ch_next, out_channels=num_conv_ch_next, kernel_size=3, padding=1),
                        th.nn.LayerNorm([num_conv_ch_next, num_conv_bins, self.num_time_frames]),
                        th.nn.GELU(),
                    ),
                ),
                th.nn.MaxPool2d(kernel_size=(2, 1)),
            ]

            num_conv_ch = num_conv_ch_next
            num_conv_ch_next *= 2
            num_conv_bins //= 2

        return conv_layers, num_conv_ch, num_conv_bins

    def forward(self, mag, phase):
        # mag.shape == phase.shape == [batch_size, num_channels, num_freq_bins, num_time_frames]

        batch_size = mag.shape[0]

        x = th.cat([mag, th.sin(phase), th.cos(phase)], dim=1)
        y = self.convolver(x)  # [batch_size, num_conv_channels, num_conv_bins, num_time_frames]

        y = y.permute(0, 3, 1, 2).reshape(batch_size, self.num_time_frames, self.conv_out_dim)

        y = self.project(y)  # [batch_size, num_time_frames, num_features]

        return y


class SkipConnection(th.nn.Module):
    def __init__(self, module: th.nn.Module):
        super().__init__()

        self.module = module

    def forward(self, x):
        return x + self.module(x)


class FeatureEncoder(th.nn.Sequential):
    def __init__(self, num_time_frames: int, num_features: int):
        num_features = int(num_features)
        num_time_frames = int(num_time_frames)

        super().__init__(
            th.nn.TransformerEncoder(
                encoder_layer=th.nn.TransformerEncoderLayer(
                    d_model=num_features,
                    dim_feedforward=4 * num_features,
                    dropout=0.15,
                    nhead=12,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=6,
            ),
        )

        self.apply(init_weights)


class TimeEncoder(th.nn.Sequential):
    def __init__(self, num_time_frames: int, num_features: int):
        self.num_time_frames = int(num_time_frames)
        self.num_features = int(num_features)

        super().__init__(
            th.nn.Linear(
                in_features=self.num_time_frames,
                out_features=self.num_features,
            ),
            th.nn.TransformerEncoder(
                encoder_layer=th.nn.TransformerEncoderLayer(
                    d_model=self.num_features,
                    dim_feedforward=4 * self.num_features,
                    dropout=0.15,
                    nhead=12,
                    activation="gelu",
                    batch_first=True,
                ),
                num_layers=6,
            ),
            th.nn.Linear(
                in_features=self.num_features,
                out_features=self.num_time_frames,
            ),
        )

        self.apply(init_weights)

    def forward(self, x):
        # x.shape == [batch_size, num_time_frames, num_features]

        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class StemExtractor(th.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        self.num_features = int(num_features)

        self.decoder = th.nn.TransformerDecoder(
            decoder_layer=th.nn.TransformerDecoderLayer(
                d_model=self.num_features,
                dim_feedforward=4 * self.num_features,
                dropout=0.15,
                nhead=12,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=6,
        )
        self.decoder.apply(init_weights)

    def forward(self, tgt, memory):
        # tgt.shape == memory.shape == [batch_size, num_time_frames, num_features]
        return self.decoder(tgt=tgt, memory=memory)


class MaskDecoder(th.nn.Module):
    def __init__(self, num_features: int, num_freq_bins: int):
        super().__init__()

        self.num_features = int(num_features)
        self.num_freq_bins = int(num_freq_bins)

        self.upscaler = th.nn.Sequential(
            th.nn.Linear(
                in_features=self.num_features,
                out_features=2 * self.num_features,
            ),
            th.nn.GELU(),
            th.nn.Dropout(p=0.15),
            th.nn.Linear(
                in_features=2 * self.num_features,
                out_features=4 * self.num_features,
            ),
            th.nn.GELU(),
            th.nn.Dropout(p=0.15),
            th.nn.Linear(
                in_features=4 * self.num_features,
                out_features=2 * self.num_freq_bins,
            ),
        )

    def forward(self, ftrs):
        # ftrs.shape == [batch_size, num_time_frames, num_features]

        mask = self.upscaler(ftrs)
        mask = th.stack([mask[:, :, :self.num_freq_bins], mask[:, :, self.num_freq_bins:]], dim=1)
        mask = mask.permute(0, 1, 3, 2)

        # mask.shape == [batch_size, num_channels, num_freq_bins, num_time_frames]
        mask = 2.0 * th.sigmoid(mask)

        return mask


class HDemucs(tha.models.HDemucs):
    def __init__(self):
        super().__init__(
            sources=["drums", "bass", "other", "vocals"],
            audio_channels=2,
        )

        self.input_length_seconds = 5.0
        self.sample_rate = 44100
        self.num_input_samples = int(self.input_length_seconds * self.sample_rate) + 1

    def forward(self, audio):
        stems = super().forward(audio.permute(0, 2, 1)).permute(0, 1, 3, 2)

        return th.stack(
            [
                stems[:, 0, :, :],
                stems[:, 1, :, :],
                stems[:, 3, :, :],
                stems[:, 2, :, :],
            ],
            dim=1,
        )
