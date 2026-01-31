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

import dataclasses
import glob
import os.path
import random
import re
import typing

import numpy as np
import torch as th

from . import effects
from . import model
from . import wav


@dataclasses.dataclass
class SongStems:
    mix: typing.List[str | np.ndarray]
    drums: typing.List[str | np.ndarray]
    bass: typing.List[str | np.ndarray]
    vocals: typing.List[str | np.ndarray]
    others: typing.List[str | np.ndarray]


def collect_songs_stems(stems_dir: str) -> typing.List[SongStems]:
    mix_re = re.compile(r"^(.*)-(mix|mixture).wav", re.IGNORECASE)
    drums_re = re.compile(r"^(.*)-drums?_*[0-9]*\.wav$", re.IGNORECASE)
    bass_re = re.compile(r"^(.*)-bass_*[0-9]*\.wav$", re.IGNORECASE)
    vocals_re = re.compile(r"^(.*)-vocals?_*[0-9]*\.wav$", re.IGNORECASE)
    others_re = re.compile(
        r"^(.*)-(accompaniment|extra|guitar|keyboard|orchestra|other|sampler|synth)s?_*[0-9]*\.wav",
        re.IGNORECASE
    )
    patterns = (
        ("mix", mix_re),
        ("drums", drums_re),
        ("bass", bass_re),
        ("vocals", vocals_re),
        ("others", others_re),
    )
    songs = {}

    dir_backlog = [stems_dir]

    while len(dir_backlog) > 0:
        dir_name = dir_backlog.pop(0)

        for entry in sorted(glob.glob(os.path.join(dir_name, "*"))):
            entry_basename = os.path.basename(entry)

            if entry_basename == "." or entry_basename == "..":
                continue

            if os.path.isdir(entry):
                dir_backlog.append(entry)

                continue

            if not entry.endswith(".wav"):
                continue

            for stem_type, ptrn in patterns:
                if mtch := ptrn.match(entry_basename):
                    song_name = mtch[1]
                    songs.setdefault(song_name, {"mix": [], "drums": [], "bass": [], "vocals": [], "others": []})
                    songs[song_name].setdefault(stem_type, []).append(entry)

    return {name: SongStems(**song) for name, song in songs.items()}


def load_song(song_info: SongStems, min_num_samples: int=0) -> tuple[SongStems, int]:
    def read_wavs(file_names: typing.List[str]) -> typing.List[np.ndarray]:
        return [read_wav(file_name) for file_name in file_names]

    def read_wav(file_name: str) -> np.ndarray:
        samples = wav.read_wav(file_name)

        if len(samples) < min_num_samples:
            samples = np.pad(samples, ((0, min_num_samples - len(samples)), (0, 0)))

        return samples

    def pad_stems(stems: typing.List[np.ndarray], length: int):
        # iterating on indices because the list will be modified in-place during the loop
        for i in range(len(stems)):
            stem = stems[i]

            if len(stem) < length:
                stems[i] = np.pad(stem, ((0, length - len(stem)), (0, 0)))

    song = SongStems(
        mix=read_wavs(song_info.mix),
        drums=read_wavs(song_info.drums),
        bass=read_wavs(song_info.bass),
        vocals=read_wavs(song_info.vocals),
        others=read_wavs(song_info.others),
    )

    all_stems = song.drums + song.bass + song.vocals + song.others

    if len(all_stems) < 1:
        raise ValueError(f"Song has no stems")

    num_samples = max(len(stem) for stem in (all_stems + song.mix))

    pad_stems(song.mix, num_samples)
    pad_stems(song.drums, num_samples)
    pad_stems(song.bass, num_samples)
    pad_stems(song.vocals, num_samples)
    pad_stems(song.others, num_samples)

    return song, num_samples


class Dataset(th.utils.data.Dataset):
    def __init__(
            self,
            songs: typing.Dict[str, SongStems],
            input_length_seconds: float=5.0,
            sample_rate: int=44100,
            num_channels: int=2,
            fft_hop_length: int=768,
    ):
        super().__init__()

        self.fft_hop_length = int(fft_hop_length)

        self.input_length_seconds, self.sample_rate, self.num_channels = model.check_params(
            input_length_seconds,
            sample_rate,
            num_channels,
        )

        self.num_input_samples = model.seconds_to_num_samples(
            self.input_length_seconds,
            self.sample_rate,
            self.fft_hop_length,
        )

        self.song_infos = []
        self.song_idx_by_name = {}
        self.song_name_by_idx = {}

        for i, (name, stems) in enumerate(songs.items()):
            self.song_idx_by_name[name] = i
            self.song_name_by_idx[i] = name
            self.song_infos.append(stems)

    def __len__(self):
        return len(self.song_infos)

    def __getitem__(self, index):
        song, num_samples, song_name = self._load_song(index)
        master, drums, bass, vocals, others = self._generate_example(song, num_samples, song_name)

        return self._example_to_input_and_target(master, drums, bass, vocals, others)

    def _load_song(self, song_idx: int) -> tuple[SongStems, int, str]:
        try:
            song, num_samples = load_song(self.song_infos[song_idx], self.num_input_samples)

        except Exception as error:
            raise RuntimeError(
                f"Error loading {self.song_name_by_idx[song_idx]!r}: {error!r}"
            ) from error

        return song, num_samples, self.song_name_by_idx[song_idx]

    def _generate_example(self, song: SongStems, num_samples: int, song_name: str) -> tuple:
        start_idx_max = max(0, num_samples - self.num_input_samples - 1)
        start_idx = random.randint(0, start_idx_max)

        drums = self._build_stem(song.drums, start_idx)
        bass = self._build_stem(song.bass, start_idx)
        vocals = self._build_stem(song.vocals, start_idx)
        others = self._build_stem(song.others, start_idx)

        master = None

        if len(song.mix) == 0:
            master = drums + bass + vocals + others
        else:
            master = self._build_stem(song.mix, start_idx)

        return master, drums, bass, vocals, others

    def _build_stem(self, stems: typing.List[np.ndarray], start_idx: int) -> np.ndarray:
        samples = np.zeros((self.num_input_samples, self.num_channels))

        for stem in stems:
            end_idx = start_idx + self.num_input_samples
            samples += stem[start_idx:end_idx, :].copy()

        return samples

    def _example_to_input_and_target(
            self,
            master: np.ndarray,
            drums: np.ndarray,
            bass: np.ndarray,
            vocals: np.ndarray,
            others: np.ndarray,
    ) -> tuple[th.Tensor, th.Tensor]:
        return (
            th.from_numpy(master),
            th.stack(
                [
                    th.from_numpy(drums),
                    th.from_numpy(bass),
                    th.from_numpy(vocals),
                    th.from_numpy(others),
                ],
            ),
        )


class TrainingDataset(Dataset):
    def __init__(
            self,
            *args,
            random_mix_ratio: float=0.3,

            pitch_prob: float=0.5,
            offset_prob: float=0.3,
            channel_swap_prob: float=0.5,
            channel_drop_prob: float=0.15,
            gain_prob: float=0.3,
            stereo_enhancer_prob: float=0.3,
            compressor_prob: float=0.3,
            dist_prob: float=0.2,
            limiter_prob: float=0.5,
            noise_prob: float=0.3,

            random_offset_max_seconds: float=0.3,

            random_eq_freq: tuple[float, float]=(80.0, 9000.0),
            random_eq_gain_db: tuple[float, float]=(-3.0, 3.0),
            random_eq_q: tuple[float, float]=(0.5, 2.0),

            random_gain_db: tuple[float, float]=(-3.0, 3.0),

            random_stereo_enhancer_width: tuple[float, float]=(0.5, 1.5),

            random_compressor_threshold_dbfs: tuple[float, float]=(-6.0, -1.0),
            random_compressor_ratio: tuple[float, float]=(1.5, 3.0),
            random_compressor_attack_seconds: tuple[float, float]=(0.01, 0.1),
            random_compressor_release_seconds: tuple[float, float]=(0.1, 0.3),

            random_dist_gain_db: tuple[float, float]=(0.0, 6.0),

            random_limiter_threshold_dbfs: tuple[float, float]=(-6.0, -3.0),
            random_limiter_limit_dbfs: tuple[float, float]=(-2.0, -0.01),
            random_limiter_attack_seconds: tuple[float, float]=(0.001, 0.05),
            random_limiter_release_seconds: tuple[float, float]=(0.05, 0.2),

            random_noise_level_dbfs: tuple[float, float]=(-48.0, -30.0),
            random_noise_hp_freq: tuple[float, float]=(20.0, 5000.0),
            random_noise_lp_freq_delta: tuple[float, float]=(3000.0, 15000.0),

            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.random_mix_ratio = float(random_mix_ratio)

        self.pitch_prob = float(pitch_prob)
        self.offset_prob = float(offset_prob)
        self.channel_swap_prob = float(channel_swap_prob)
        self.channel_drop_prob = float(channel_drop_prob)
        self.gain_prob = float(gain_prob)
        self.stereo_enhancer_prob = float(stereo_enhancer_prob)
        self.compressor_prob = float(compressor_prob)
        self.dist_prob = float(dist_prob)
        self.limiter_prob = float(limiter_prob)
        self.noise_prob = float(noise_prob)

        self.random_offset_max_seconds = float(random_offset_max_seconds)
        self.random_offset_max = int(self.random_offset_max_seconds * self.sample_rate)

        self.random_eq_freq_min, self.random_eq_freq_max = random_eq_freq
        self.random_eq_gain_db_min, self.random_eq_gain_db_max = random_eq_gain_db
        self.random_eq_q_min, self.random_eq_q_max = random_eq_q

        self.random_gain_db_min, self.random_gain_db_max = random_gain_db

        self.random_stereo_enhancer_width_min, self.random_stereo_enhancer_width_max = random_stereo_enhancer_width

        self.random_compressor_threshold_dbfs_min, self.random_compressor_threshold_dbfs_max = random_compressor_threshold_dbfs
        self.random_compressor_ratio_min, self.random_compressor_ratio_max = random_compressor_ratio
        self.random_compressor_attack_seconds_min, self.random_compressor_attack_seconds_max = random_compressor_attack_seconds
        self.random_compressor_release_seconds_min, self.random_compressor_release_seconds_max = random_compressor_release_seconds

        self.random_dist_gain_db_min, self.random_dist_gain_db_max = random_dist_gain_db

        self.random_limiter_threshold_dbfs_min, self.random_limiter_threshold_dbfs_max = random_limiter_threshold_dbfs
        self.random_limiter_limit_dbfs_min, self.random_limiter_limit_dbfs_max = random_limiter_limit_dbfs
        self.random_limiter_attack_seconds_min, self.random_limiter_attack_seconds_max = random_limiter_attack_seconds
        self.random_limiter_release_seconds_min, self.random_limiter_release_seconds_max = random_limiter_release_seconds

        self.random_noise_level_dbfs_min, self.random_noise_level_dbfs_max = random_noise_level_dbfs
        self.random_noise_hp_freq_min, self.random_noise_hp_freq_max = random_noise_hp_freq
        self.random_noise_lp_freq_delta_min, self.random_noise_lp_freq_delta_max = random_noise_lp_freq_delta

        num_random_mixes = max(0, int(len(self.song_infos) * self.random_mix_ratio))
        max_song_idx = len(self.song_infos) - 1

        for i in range(num_random_mixes):
            stems = {"drums": [], "bass": [], "vocals": [], "others": []}

            for stem_type in stems.keys():
                complexity = random.randint(1, 2)

                for j in range(complexity):
                    found_one = False
                    random_song_stems = []

                    while not found_one:
                        random_song = self.song_infos[random.randint(0, max_song_idx)]
                        random_song_stems = getattr(random_song, stem_type)
                        found_one = len(random_song_stems) > 0

                    stem_idx = random.randint(0, len(random_song_stems) - 1)
                    stems[stem_type].append(random_song_stems[stem_idx])

            idx = max_song_idx + i + 1
            name = f"random-{i:03}"
            self.song_idx_by_name[name] = idx
            self.song_name_by_idx[idx] = name
            stems["mix"] = []
            self.song_infos.append(SongStems(**stems))

    def _generate_example(self, song: SongStems, num_samples: int, song_name: str) -> tuple:
        num_input_samples = self.num_input_samples

        if random.random() < self.pitch_prob:
            num_input_samples = int(self.num_input_samples * (0.7 + 0.5 * random.random()))

        start_idx_max = max(0, num_samples - num_input_samples - 1)
        start_idx = random.randint(0, start_idx_max)

        drums = self._build_stem(song.drums, start_idx, num_input_samples)
        bass = self._build_stem(song.bass, start_idx, num_input_samples)
        vocals = self._build_stem(song.vocals, start_idx, num_input_samples)
        others = self._build_stem(song.others, start_idx, num_input_samples)

        effects_cfg = self._build_random_eq_config()
        gain_db_est = 0.0

        gain_db_est += self._add_random_gain(effects_cfg)
        gain_db_est += self._add_random_stereo_enhancer(effects_cfg)
        gain_db_est += self._add_random_compressor(effects_cfg)
        gain_db_est += self._add_random_dist(effects_cfg)
        gain_db_est += self._add_random_limiter(effects_cfg)
        gain_db_est += self._add_random_noise(effects_cfg)

        effects_cfg.append(("wave_shaper", {"curve": "clipper"}))

        mix = drums + bass + vocals + others
        master = mix.copy()

        effects.apply_effects(master, effects_cfg)

        mix_rms = np.sqrt((mix * mix).mean())
        master_rms = np.sqrt((master * master).mean())

        gain_db = gain_db_est

        if mix_rms > 1e-6 and master_rms > 1e-6:
            # Compression and other non-linearities may affect the stems
            # differently, and they can distort both the measured and the
            # estimated loudness change in different ways. Hopefully,
            # combining the two will reduce the overall error.

            gain_db_mes = effects.linear_to_db(master_rms / mix_rms)
            gain_db = (gain_db_est + gain_db_mes) / 2.0

        gain_cfg = [("gain", {"gain_db": gain_db_est})]

        effects.apply_effects(drums, gain_cfg)
        effects.apply_effects(bass, gain_cfg)
        effects.apply_effects(vocals, gain_cfg)
        effects.apply_effects(others, gain_cfg)

        return master, drums, bass, vocals, others

    def _build_stem(
            self,
            stems: typing.List[np.ndarray],
            start_idx: int,
            num_input_samples: int,
    ) -> np.ndarray:
        samples = np.zeros((num_input_samples, self.num_channels))

        for stem in stems:
            offset_idx = 0

            if random.random() < self.offset_prob:
                offset_idx = random.randint(0, min(start_idx, self.random_offset_max))

            rnd_start_idx = start_idx - offset_idx
            end_idx = rnd_start_idx + num_input_samples
            component = stem[rnd_start_idx:end_idx, :].copy()

            if random.random() < self.channel_swap_prob:
                component = np.stack((component[:, 1], component[:, 0])).T

            for channel_idx in range(self.num_channels):
                if random.random() < self.channel_drop_prob:
                    component[:, channel_idx] = 0.0

            gain_db = self._random(self.random_gain_db_min, self.random_gain_db_max)
            effects.apply_effects(
                component,
                [("gain", {"gain_db": gain_db})] + self._build_random_eq_config()
            )
            samples += component

        return self._interpolate(samples, num_input_samples)

    def _interpolate(self, samples: np.ndarray, num_input_samples: int) -> np.ndarray:
        if num_input_samples == self.num_input_samples:
            return samples

        idx = np.linspace(0, num_input_samples - 1, self.num_input_samples)
        idx_low = np.floor(idx).astype(int)
        idx_high = idx_low + 1
        idx_high[-1] -= 1
        weights = (idx - idx_low).reshape(self.num_input_samples, 1)

        return (1.0 - weights) * samples[idx_low] + weights * samples[idx_high]

    @staticmethod
    def _random(min: float, max: float) -> float:
        return random.random() * (max - min) + min

    def _build_random_eq_config(self) -> typing.List[tuple]:
        eq_cfg = []
        eq_num = random.randint(0, 3)

        for i in range(eq_num):
            freq = self._random(self.random_eq_freq_min, self.random_eq_freq_max)
            gain_db = self._random(self.random_eq_gain_db_min, self.random_eq_gain_db_max)
            q = self._random(self.random_eq_q_min, self.random_eq_q_max)

            eq_cfg.append(("bell", {"freq": freq, "q": q, "gain_db": gain_db}))

        return eq_cfg

    def _add_random_gain(self, effects_cfg: list) -> float:
        if random.random() >= self.gain_prob:
            return 0.0

        gain_db = self._random(self.random_gain_db_min, self.random_gain_db_max)
        effects_cfg.append(("gain", {"gain_db": gain_db}))

        return gain_db

    def _add_random_stereo_enhancer(self, effects_cfg: list) -> float:
        if random.random() >= self.stereo_enhancer_prob:
            return 0.0

        width = self._random(
            self.random_stereo_enhancer_width_min,
            self.random_stereo_enhancer_width_max
        )
        effects_cfg.append(("stereo_enhancer", {"width": width}))

        return 2.0 * (width - 1.0)

    def _add_random_compressor(self, effects_cfg: list) -> float:
        if random.random() >= self.compressor_prob:
            return 0.0

        threshold_dbfs = self._random(
            self.random_compressor_threshold_dbfs_min,
            self.random_compressor_threshold_dbfs_max,
        )
        ratio = self._random(
            self.random_compressor_ratio_min,
            self.random_compressor_ratio_max,
        )
        attack_seconds = self._random(
            self.random_compressor_attack_seconds_min,
            self.random_compressor_attack_seconds_max,
        )
        release_seconds = self._random(
            self.random_compressor_release_seconds_min,
            self.random_compressor_release_seconds_max,
        )
        make_up_gain_db = max(0.0, - threshold_dbfs * (1.0 - 1.0 / ratio) - 1.0)

        effects_cfg.append(
            (
                "compressor",
                {
                    "threshold_dbfs": threshold_dbfs,
                    "ratio": ratio,
                    "attack_seconds": attack_seconds,
                    "release_seconds": release_seconds,
                },
            )
        )

        if make_up_gain_db > 0.0:
            effects_cfg.append(("gain", {"gain_db": make_up_gain_db}))

        return make_up_gain_db

    def _add_random_dist(self, effects_cfg: list) -> float:
        if random.random() >= self.dist_prob:
            return 0.0

        dist_gain_db = self._random(self.random_dist_gain_db_min, self.random_dist_gain_db_max)
        effects_cfg.append(("gain", {"gain_db": dist_gain_db}))
        effects_cfg.append(("wave_shaper", {"curve": "tanh"}))

        return dist_gain_db / 3.0

    def _add_random_limiter(self, effects_cfg: list) -> float:
        if random.random() >= self.limiter_prob:
            return 0.0

        threshold_dbfs = self._random(
            self.random_limiter_threshold_dbfs_min,
            self.random_limiter_threshold_dbfs_max,
        )
        limit_dbfs = self._random(
            self.random_limiter_limit_dbfs_min,
            self.random_limiter_limit_dbfs_max,
        )
        attack_seconds = self._random(
            self.random_limiter_attack_seconds_min,
            self.random_limiter_attack_seconds_max,
        )
        release_seconds = self._random(
            self.random_limiter_release_seconds_min,
            self.random_limiter_release_seconds_max,
        )
        effects_cfg.append(
            (
                "limiter",
                {
                    "threshold_dbfs": threshold_dbfs,
                    "limit_dbfs": limit_dbfs,
                    "attack_seconds": attack_seconds,
                    "release_seconds": release_seconds,
                },
            )
        )

        return limit_dbfs - threshold_dbfs

    def _add_random_noise(self, effects_cfg: list) -> float:
        if random.random() >= self.noise_prob:
            return 0.0

        noise_level_dbfs = self._random(
            self.random_noise_level_dbfs_min,
            self.random_noise_level_dbfs_max,
        )
        noise_hp = self._random(self.random_noise_hp_freq_min, self.random_noise_hp_freq_max)
        noise_lp = noise_hp + self._random(
            self.random_noise_lp_freq_delta_min,
            self.random_noise_lp_freq_delta_max,
        )
        effects_cfg.append(
            (
                "add_noise",
                {
                    "level_dbfs": noise_level_dbfs,
                    "high_pass_freq": noise_hp,
                    "low_pass_freq": noise_lp,
                },
            )
        )

        return 0.0
