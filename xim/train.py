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

import math
import time
import typing

import torch as th


class TrainingLoop:
    def __init__(
            self,
            device: th.device,
            dtype: th.dtype,
            model: th.nn.Module,
            train_dl: th.utils.data.DataLoader,
            test_dl: th.utils.data.DataLoader,
            optimizer: th.optim.Optimizer,
            num_epochs: int,
            first_epoch_idx: int=0,
            num_grad_acc_batches: int=1,
            lr_scheduler: typing.Optional[th.optim.lr_scheduler.LRScheduler]=None,
    ):
        self.device = device
        self.dtype = dtype
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optimizer
        self.num_epochs = int(num_epochs)
        self.first_epoch_idx = int(first_epoch_idx)
        self.num_grad_acc_batches = int(num_grad_acc_batches)
        self.lr_scheduler = lr_scheduler

        self.train_losses = None
        self.test_losses = None

    def loss(self, pred: th.Tensor, target: th.Tensor) -> th.Tensor:
        raise NotImplementedError("Loss function must be implemented")

    def run(self) -> tuple[typing.List[float], typing.List[float]]:
        self.model.to(self.dtype).to(self.device)

        self.init()

        self.optimizer.zero_grad()

        for epoch_idx in range(self.first_epoch_idx, self.num_epochs):
            epoch_start = time.time()

            model_extra_args = self.before_epoch(epoch_idx) or ()

            self.model.train(True)
            train_loss, train_num_batches = self.run_epoch(
                epoch_idx,
                "train",
                self.train_dl,
                model_extra_args,
            )
            self.train_losses.append(train_loss)

            self.model.train(False)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            with th.no_grad():
                test_loss, test_num_batches = self.run_epoch(
                    epoch_idx,
                    "test",
                    self.test_dl,
                    model_extra_args,
                )
                self.test_losses.append(test_loss)

            epoch_end = time.time()
            self.after_epoch(epoch_idx, epoch_start, epoch_end, train_num_batches, test_num_batches)

        return self.train_losses, self.test_losses

    def init(self):
        self.train_losses = []
        self.test_losses = []

    def before_epoch(self, epoch_idx: int) -> typing.Optional[tuple]:
        return None

    def after_epoch(
            self,
            epoch_idx: int,
            epoch_start: float,
            epoch_end: float,
            train_num_batches: int,
            test_num_batches: int,
    ):
        print(self.epoch_loss_to_str(epoch_idx, epoch_start, epoch_end, train_num_batches, test_num_batches))

    def epoch_loss_to_str(
            self,
            epoch_idx: int,
            epoch_start: float,
            epoch_end: float,
            train_num_batches: int,
            test_num_batches: int,
    ) -> str:
        time_delta = epoch_end - epoch_start

        train_loss = self.train_losses[-1]
        test_loss = self.test_losses[-1]

        train_loss_min_str = "..."
        test_loss_min_str = "..."

        if len(self.train_losses) < 2 or train_loss < min(self.train_losses[:-1]):
            train_loss_min_str = "MIN"

        if len(self.test_losses) < 2 or test_loss < min(self.test_losses[:-1]):
            test_loss_min_str = "MIN"

        return (
            f"Epoch {epoch_idx + 1:>5}/{self.num_epochs:<5}"
            f" {time_delta:>9.3f}s"
            f"   Loss: {train_loss:.6f} / {test_loss:.6f}"
            f" {train_loss_min_str} / {test_loss_min_str}"
        )

    def run_epoch(
            self,
            epoch_idx: int,
            stage: str,
            data_loader: th.utils.data.DataLoader,
            model_extra_args: tuple
    ) -> tuple[float, int]:
        loss = 0.0

        for i, (input_, target) in enumerate(data_loader):
            input_ = input_.to(self.dtype).to(self.device)
            target = target.to(self.dtype).to(self.device)

            self.before_batch(epoch_idx, stage, i, input_, target)

            pred = self.model(input_, *model_extra_args)
            batch_loss = self.loss(pred, target)

            loss += batch_loss.item()

            if stage == "train" and th.isfinite(batch_loss):
                batch_loss.backward()

                if (i + 1) % self.num_grad_acc_batches == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            model_extra_args = (
                self.after_batch(epoch_idx, stage, i, batch_loss, input_, pred, target) or ()
            )

        num_batches = i + 1

        return loss / num_batches, num_batches

    def before_batch(
            self,
            epoch_idx: int,
            stage: str,
            batch_idx: int,
            input_: th.Tensor,
            target: th.Tensor,
    ):
        pass

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
        pass
