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
import traceback
import typing

import torch as th

import xim.data
import xim.effects
import xim.model
import xim.train
import xim.wav


def main(argv):
    try:
        train_dir_name = argv[1]
        test_dir_name = argv[2]
        snapshot_dir_name = argv[3]

    except Exception as error:
        print(
            f"Usage: {os.path.basename(argv[0])} train_dir test_dir snapshot_dir [xim|hdemucs] [lr] [epoch/nepochs] [model.pt]",
            file=sys.stderr,
        )
        print(f"{type(error)}: {error}")

        return 1

    model_name = argv[4] if len(argv) > 4 else "xim"
    learning_rate = float(argv[5]) if len(argv) > 5 else 1e-4
    epoch_cfg = argv[6] if len(argv) > 6 else "1/10"
    model_weights_file_name = argv[7] if len(argv) > 7 else None

    first_epoch_idx, num_epochs = [int(part) for part in epoch_cfg.split("/", 1)]
    first_epoch_idx -= 1

    batch_size = 1
    num_grad_acc_batches = 32 // batch_size

    weight_decay = 0.01 / num_grad_acc_batches
    learning_rate_decay_factor = 1.0

    device_name = "cuda" if th.cuda.is_available() else "cpu"

    device = th.device(device_name)
    model = xim.model.create_model(model_name).to(device)

    print(f"Starting training:")
    print(f"  {model_name=!r}")
    print(f"  epochs={first_epoch_idx + 1}/{num_epochs}")
    print(f"  lr={learning_rate:.9f}")
    print(f"  init_weights={model_weights_file_name!r}")
    print(f"  {batch_size=}")
    print(f"  {num_grad_acc_batches=}")
    print(f"  {weight_decay=:.9f}")
    print(f"  lr_decay_factor={learning_rate_decay_factor:.9f}")
    print(f"  {device_name=!r}")

    print(model)

    model.train(True)

    if model_weights_file_name:
        model.load_state_dict(
            th.load(
                model_weights_file_name,
                weights_only=True,
                map_location=device,
            ),
        )

    train_songs = xim.data.collect_songs_stems(train_dir_name)
    test_songs = xim.data.collect_songs_stems(test_dir_name)

    train_ds = xim.data.TrainingDataset(train_songs)
    test_ds = xim.data.Dataset(test_songs)

    train_dl = th.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
    )
    test_dl = th.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = th.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    lr_scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda epoch_idx: max(0.03, learning_rate_decay_factor ** epoch_idx)
    )

    gc.collect()
    th.cuda.empty_cache()

    training_loop = TrainingLoop(
        model_name=model_name,
        snapshot_dir_name=snapshot_dir_name,
        device=device,
        dtype=th.float,
        model=model,
        train_dl=train_dl,
        test_dl=test_dl,
        optimizer=optimizer,
        num_epochs=num_epochs,
        first_epoch_idx=first_epoch_idx,
        num_grad_acc_batches=num_grad_acc_batches,
        lr_scheduler=lr_scheduler,
    )
    train_losses, test_losses = training_loop.run()

    return 0


class TrainingLoop(xim.train.TrainingLoop):
    def __init__(self, model_name: str, snapshot_dir_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_name = model_name
        self.snapshot_dir_name = snapshot_dir_name

        self.batch_loss_avg_window_length = max(10, self.num_grad_acc_batches * 10)
        self.batch_losses = {}

    def loss(self, pred: th.Tensor, target: th.Tensor) -> th.Tensor:
        mse = th.nn.functional.mse_loss(pred, target)
        l1 = th.nn.functional.l1_loss(pred, target)

        return mse + l1

    def before_epoch(self, epoch_idx: int) -> typing.Optional[tuple]:
        pass

    def after_epoch(
            self,
            epoch_idx: int,
            epoch_start: float,
            epoch_end: float,
            train_num_batches: int,
            test_num_batches: int,
    ):
        super().after_epoch(
            epoch_idx,
            epoch_start,
            epoch_end,
            train_num_batches,
            test_num_batches,
        )

        self.save_snapshot(epoch_idx)

    def save_snapshot(self, epoch_idx: int):
        try:
            file_name = os.path.join(self.snapshot_dir_name, f"{self.model_name}-{epoch_idx:03}.pt")
            print(f"  Saving snapshot: {file_name}")
            th.save(self.model.state_dict(), file_name)

        except:
            print(f"Error saving model state; {epoch_idx=}")
            print(traceback.format_exc())

    def after_batch(
            self,
            epoch_idx: int,
            stage: str,
            batch_idx: int,
            batch_loss: th.Tensor,
            input_: th.Tensor,
            pred: th.Tensor,
            target: th.Tensor,
    ) -> typing.Optional[tuple]:

        if batch_idx % 300 == 0:
            self.save_snapshot(epoch_idx)

        batch_losses = self.batch_losses.setdefault(stage, [])
        batch_losses.append(batch_loss.item())

        if batch_idx % 10 == 0:
            batch_losses = batch_losses[-self.batch_loss_avg_window_length:]
            self.batch_losses[stage] = batch_losses
            mean = sum(batch_losses) / len(batch_losses)

            weights = th.nn.utils.parameters_to_vector(self.model.parameters())
            weights_mean = weights.abs().mean()
            weights_std = weights.std()

            print(
                f"  {stage:<6} {batch_idx + 1:>5}  L:{batch_loss:.6f}"
                f"  Avg:{mean:.6f}"
                f"  E|W|={weights_mean:.6f}"
                f"  stdW={weights_std:.6f}"
            )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
