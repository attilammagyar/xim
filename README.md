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

 * Split [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html) stems
   into smaller chunks (so that the randomized training data set won't have to
   load entire tracks just for taking short samples), and remove any metronome
   marks and other irrelevant parts from the beginning of some multitracks:

       python3 split_stems.py raw_stems/start_times.tsv raw_stems/ stems/

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

Data augmentation
-----------------

The music source separation (MSS) problem only seeks to separate the
stems that sum up to the given mix, but most real-world music recordings are
also treated with various non-linearities during the
[mastering](https://en.wikipedia.org/wiki/Mastering_(audio)) process. In the
hope that it might help the model learn to ignore the artifacts associated with
these effects, the `TrainingDataset` class in `xim/data.py` uses a random
combination of mastering effects and other augmentation techniques, including:

 * pitch shifting,
 * slight timing variations,
 * mixing stems of random songs,
 * [equalization](https://en.wikipedia.org/wiki/Audio_equalization),
 * soft-clipping (implemented with
   [antiderivative anti-aliasing wave shapers, a.k.a. ADAA](https://en.wikipedia.org/wiki/Mastering_(audio)))
 * stereo widening,
 * dynamic range compression,
 * brickwall-limiting with random threshold,
 * etc.

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

 * pre-trained HDemucs in [Torchaudio](https://docs.pytorch.org/audio/stable/index.html):

    * L1 = 0.011
    * SDR = 9.660 dB

 * pre-trained HDemucs tuned on MUSDB18-HQ and 122 other multitracks with
   random mastering for 2 epochs with `1e-5` learning rate:

    * L1 = 0.012
    * SDR = 9.496 dB

 * Xim trained from scratch on MUSDB18-HQ and 122 other multitracks with
   random mastering for 12 epochs with `1e-5` learning rate:

    * L1 = 0.028
    * SDR = 2.187 dB

Maybe a normal unmixer model and a specific unmastering model would be a better
way to approach this?
