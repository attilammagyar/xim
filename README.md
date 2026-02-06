Xim
===

(Work-in-progress) My first attempt at music unmastering and unmixing.

This is a work-in-progress pet project - if you found this repo looking for
actually useful music unmixers, try one of these:

 * [HDemucs](https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html),
 * [HTDemucs](https://github.com/adefossez/demucs),
 * [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer),
 * or simply use the built-in track splitter tool in [FL Studio](https://www.image-line.com/)
   (it works with the demo version as well).

Cheat sheet
-----------

 * Compile `effects.pyx` and run its tests:

       python3 setup.py build_ext --inplace
       python3 -m unittest

 * Split stems into smaller chunks (so that the randomized training data set
   won't have to load entire tracks just for taking short samples), and remove
   any metronome marks and other irrelevant parts from the beginning of some
   multitracks:

       python3 split_stems.py raw_stems/start_times.tsv raw_stems/ stems/

   The `raw_stems` directory is processed recursively. Files are expected to
   have names like `some_song-guitars.wav`, `some_song-drums.wav`, etc.

 * Train a model from scratch with various learning rates:

       python3 train.py ./stems/train/ ./stems/test/ ./snapshots xim 1e-5 1/12
       python3 train.py ./stems/train/ ./stems/test/ ./snapshots hdemucs 1e-4 1/12

   (Note: HDemucs is fine with `1e-4`, Xim learns better with `1e-5`.)

 * Fine-tune a pretrained [HDemucs](https://docs.pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html)
   model after [downloading the weights](https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt):

       python3 train.py ./stems/train/ ./stems/test/ ./snapshots hdemucs 1e-5 1/3 hdemucs_high_trained.pt

 * Separate a 44.1 kHz, 16 bit stereo WAV file:

       python3 separate.py hdemucs_high_trained.pt song.wav ./stems/ hdemucs

 * Evaluate a model on e.g. the MUSDB18-HQ dataset:

       python3 eval.py xim.pt raw_stems/musdb18hq/test/ xim
       python3 eval.py hdemucs_high_trained.pt raw_stems/musdb18hq/test/ hdemucs

Problem
-------

 * Traditional music source separation (MSS) focuses on separating audio tracks
   (stems) that sum up to produce the given mix.

 * Real-world music recordings are usually treated with a number of
   non-linearities during a process called
   [mastering](https://en.wikipedia.org/wiki/Mastering_(audio).

    * E.g. dynamic range compression and brickwall limiting: when the track
      gets loud (e.g. kick drum hit), its loudness is reduced for a split
      second so that it cannot go above a certain level - then the loudness of
      the entire track is raised. (See: [loudness war](https://en.wikipedia.org/wiki/Loudness_war).)

   Separating a mastered track often results in stems with noticable
   short term loudness variations ("pumping").

Data Augmentation
-----------------

In the hope that it might help the model learn to ignore mastering artifacts,
the `TrainingDataset` class in `xim/data.py` uses a random combination of
traditional MSS augmentation techniques and mastering effects, including:

 * pitch shifting,
 * slight timing variations,
 * mixing stems of random songs,
 * silencing stems or stereo channels,
 * [equalization](https://en.wikipedia.org/wiki/Audio_equalization),
 * soft-clipping (implemented with
   [antiderivative anti-aliasing wave shapers, a.k.a. ADAA](https://en.wikipedia.org/wiki/Mastering_(audio)))
 * stereo widening,
 * dynamic range compression,
 * brickwall-limiting,
 * etc.

(Most of these were implemented in [Cython](https://cython.org/) and C++.)

Data
----

 * [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html)
 * 122 other other multitracks from various sources (e.g. FL Studio demo
   projects rendered to separate stems, etc.)

Challenges
----------

 * It is hard to come by multitrack recordings with all instruments separated.
   A common compromise is to separate 4 stems: drums, bass, vocals, and others.

    * Stems recorded to tape often contain multiple instruments bounced to the
      same track due to the limited number of available tape tracks.

 * Bleeding:

    * Instruments often bleed into vocal stems through the vocalist's
      headphones.

    * Live recorded tracks often bleed into each other's microphones in the
      room.

 * Some stems don't just have bleeding, they contain entire parts that should
   be elsewhere.

 * It is hard to tell exactly how mastering affects the overall loudness of
   individual stems, therefore the ground truth is sometimes vague.

 * In some genres like [EDM](https://en.wikipedia.org/wiki/Electronic_dance_music),
   the pumping effect is an intentional stylistic choice.
   (See also: [side-chain compression](https://en.wikipedia.org/wiki/Dynamic_range_compression#Side-chaining).)

Models
-----

This project contains 2 models:

### HDemucs

 * Built into [Torchaudio](https://docs.pytorch.org/audio/stable/index.html),
   used for comparison and experimenting.

 * Uses a U-Net-like convolutional architecture to process both the raw
   waveform and the spectrogram (hence Hybrid Demucs).

### Xim

 * The model expects 5 seconds long audio snippets.

    * Longer audio can be processed by splitting it into smaller chunks with
      some overlap, then merging the results with cross-fade.

 * Performs [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
   with overlapping windows, converting the magnitudes to log-scale.

 * 3x3 convolutions followed by a linear projection compress the resulting
   magnitude and phase information along the frequency bins axis into a
   manageable number of features, keeping the resolution of the time axis.

 * A pair of transformer encoders independently analyzes the resulting 2D image
   (after adding a learnable positional embedding) along perpendicular axes:

    * along the time axis to extract information about the relationships of
      the features over time,

    * along the feature dimensions axis to extract information about harmonic
      relationships.

 * The results of the encoders are summed up and passed to a sequence of
   transformer decoders as `memory`, along with the positionally encoded
   sequence of features as target.

 * Each decoder is responsible for a single stem: a decoder produces a sequence
   of feature deltas which get subtracted from the feature sequence before
   being passed to the next decoder.

    * The idea here is that the feature space should contain multiple versions
      of the same recording, and each decoder should figure out how to get
      from one version to the next:

       1. full mix,
       2. drumless version,
       3. drum and bass removed version,
       4. a single "others" track,
       5. and finally, complete silence.

   Since the output of the last decoder is supposed to lead to complete
   silence, the "others" track has no decoder layers, just the mask generator
   network.

 * A feature delta produced by a decoder is converted into a mask by a
   decoder-specific shallow dense network and the sigmoid function.

    * The mask can take values between 0 and 2: when the STFT magnitudes are
      multiplied elementwise with a mask, this allows the model to increase
      some of the magnitudes, making it possible to remove some mastering
      artifacts.

 * Finally, the masked magnitudes and the original phase are turned back into
   audio using inverse STFT.

Training
--------

I used MSE + L1 for the loss function. (MSE alone is not recommended for MSS.)

For regularization, I used 15% dropout.

MUSDB18-HQ Results (WIP)
------------------------

(L1: sum of the absolute value of differences between the true stem and the
model's prediction; lower is better. SDR: signal to distortion ratio in
[dB](https://en.wikipedia.org/wiki/Decibel), see
[AIcrowd/music-demixing-challenge-starter-kit](https://github.com/AIcrowd/music-demixing-challenge-starter-kit);
higher is better.)

 * HDemucs trained from scratch on MUSDB18-HQ and 122 other multitracks with
   random mastering for 3 epochs with `1e-4` learning rate:

    * L1 = 0.025
    * SDR = 3.042 dB

 * pre-trained HDemucs:

    * L1 = 0.011
    * SDR = 9.660 dB

 * pre-trained HDemucs tuned on MUSDB18-HQ and 122 other multitracks with
   random mastering for 2 epochs with `1e-5` learning rate:

    * L1 = 0.012
    * SDR = 9.496 dB

 * Xim trained from scratch on MUSDB18-HQ and 122 other multitracks with
   random mastering for 12 epochs with `1e-5` learning rate:

    * L1 = 0.026
    * SDR = 2.671 dB

 * Xim trained from scratch on MUSDB18-HQ and 122 other multitracks with
   random mastering for 93 epochs with `1e-5` learning rate:

    * L1 = 0.023
    * SDR = 3.449 dB

Gotchas
-------

 * It takes up to 3-5 epochs for the model to realize that the time axis
   exists. Up until then, it will produce masks with almost no variance over
   the time axis, reducing the whole operation into a collection of
   computationally very expensive [band-pass filters](https://en.wikipedia.org/wiki/Band-pass_filter).

 * Good things come to those who wait: the loss explodes after a few epochs if
   if the learning rate is too large; `1e-5` seems fine.

Conclusion, further development
-------------------------------

Maybe a separate unmixer and a specific unmastering model would be a better
way to approach this? (This would solve the ground truth ambiguity.)
