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

import os
import os.path
import sys

import numpy as np

import xim.data
import xim.effects
import xim.wav


SAMPLE_RATE = 44100
CHUNK_LENGTH_SECONDS = 10.0
CHUNK_OVERLAP_SECONDS = 0.5
FADE_IN_SECONDS = 0.05
FADE_OUT_SECONDS = 0.1
SILENCE_THRESHOLD_DB = -60.0


def main(argv):
    """
    Split raw music multitrack stems into short chunks for more efficient I/O
    usage during training.

    The raw stems directory is expected to contain files named like:
    `artist_name-song_title-guitars_1.wav`, `artist_name-song_title-guitars_2.wav`,
    `artist_name-song_title-vocals.wav`, etc.

    The `start_times.tsv` file should be a tab separated list of start times
    and `artist_name-song_title` file names. This TSV file should tell how many
    seconds to discard from the beginning of each multitrack song, e.g. due to
    containing silence, metronome ticks, etc. Example:

    1.5<TAB>artist_name-song_title
    2.3<TAB>other_artist-other_song

    The raw stems are expected to be aligned with each other and fit together.
    (Note that some multitracks circulating on the web seem to contain stems
    which might have been transferred from tape in multiple passes or in
    different sessions, so even though they belong to the same recording, when
    they are played back together, they gradually go more and more out-of-sync,
    probably due to tape wow and flutter.)
    """

    if len(argv) < 4:
        print(f"Usage: {os.path.basename(argv[0])} start_times.tsv raw_stems_dir chunks_dir")

        return 1

    start_times_tsv = argv[1]
    raw_stems_dir = argv[2]
    chunks_dir = argv[3]

    chunk_num_samples = int(CHUNK_LENGTH_SECONDS * SAMPLE_RATE) + 3
    chunk_overlap_num_samples = int(CHUNK_OVERLAP_SECONDS * SAMPLE_RATE) + 1

    fade_in_num_samples = int(FADE_IN_SECONDS * SAMPLE_RATE) + 1
    fade_in = np.linspace(0.0, 1.0, fade_in_num_samples)
    fade_in = np.stack([fade_in, fade_in], axis=1)

    fade_out_num_samples = int(FADE_OUT_SECONDS * SAMPLE_RATE) + 1
    fade_out = np.linspace(1.0, 0.0, fade_out_num_samples)
    fade_out = np.stack([fade_out, fade_out], axis=1)

    silence_threshold = xim.effects.db_to_linear(SILENCE_THRESHOLD_DB)
    songs = xim.data.collect_songs_stems(raw_stems_dir)
    start_times = {}

    with open(start_times_tsv, "r") as f:
        for line in f:
            start_time, song_name = line.strip().split("\t", 1)
            start_times[song_name] = float(start_time)

    for i, (song_name, song) in enumerate(songs.items()):
        print(f"{song_name}")

        stems_by_type = [
            ("mix", song.mix),
            ("drums", song.drums),
            ("bass", song.bass),
            ("vocals", song.vocals),
            ("others", song.others),
        ]
        stems = []

        start_samples = int(start_times.get(song_name, 0.0) * SAMPLE_RATE)

        for stem_type, stem_file_names in stems_by_type:
            for j, stem_file_name in enumerate(stem_file_names):
                stem_name = f"{stem_type}_{j}"
                stem_samples = xim.wav.read_wav(stem_file_name)
                stem_samples = stem_samples[start_samples:, :]
                stem_samples[0:fade_in_num_samples] *= fade_in
                stem_samples = chop_trailing_silence(stem_samples, silence_threshold)
                stem_samples[-fade_out_num_samples:] *= fade_out
                stems.append((song_name, stem_name, stem_samples))

        max_length = max(len(stem_samples) for song_name, stem_name, stem_samples in stems)

        stems = [
            (song_name, stem_name, np.pad(stem_samples, ((0, max_length - len(stem_samples)), (0, 0))))
            for song_name, stem_name, stem_samples in stems
        ]

        song_dir = os.path.join(chunks_dir, song_name)
        os.makedirs(song_dir, exist_ok=True)

        start_idx = 0
        last_split_after_idx = max_length - chunk_num_samples * 2
        j = 0

        while start_idx < last_split_after_idx:
            end_idx = start_idx + chunk_num_samples

            for song_name, stem_name, stem_samples in stems:
                chunk_file_name = os.path.join(song_dir, f"{song_name}-{j:03}-{stem_name}.wav")

                print(f"  {os.path.basename(chunk_file_name)}")

                chunk = stem_samples[start_idx:end_idx, :]
                xim.wav.write_wav(chunk_file_name, chunk)

            j += 1
            start_idx += chunk_num_samples - chunk_overlap_num_samples

        for song_name, stem_name, stem_samples in stems:
            chunk_file_name = os.path.join(song_dir, f"{song_name}-{j:03}-{stem_name}.wav")

            print(f"  {os.path.basename(chunk_file_name)}")

            chunk = stem_samples[start_idx:, :]
            xim.wav.write_wav(chunk_file_name, chunk)

    return 0


def chop_trailing_silence(samples: np.ndarray, silence_threshold) -> np.ndarray:
    silence_start_idx = np.flatnonzero(np.max(np.abs(samples), axis=1) > silence_threshold)[-1]

    return samples[0:silence_start_idx + 1, :]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
