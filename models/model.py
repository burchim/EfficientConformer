# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Sentencepiece
import sentencepiece as spm

# Schedulers
from models.schedules import *

# Other
from tqdm import tqdm
import jiwer
import os
import time

def sample_synaptic_noise(m, distributed):

    if hasattr(m, "sample_synaptic_noise"):
        m.sample_synaptic_noise(distributed)

def init_vn(m, vn_std):

    if hasattr(m, "init_vn"):
        m.init_vn(vn_std)

class Model(nn.Module):

    def __init__(self, tokenizer_params, training_params, decoding_params, name):
        super(Model, self).__init__()

        # Tokenizer
        try:
            self.tokenizer = spm.SentencePieceProcessor(tokenizer_params["tokenizer_path"])
        except:
            self.tokenizer = None
            print("Tokenizer not found...")

        # Training Params
        self.encoder_frozen_steps = training_params.get("encoder_frozen_steps", None)
        self.vn_start_step = training_params.get("vn_start_step", None)

        # Decoding Params
        self.beam_size = decoding_params.get("beam_size", 1)
        self.tmp = decoding_params.get("tmp", 1)

        # Ngram
        self.ngram_path = decoding_params.get("ngram_path", None)
        self.ngram_alpha = decoding_params.get("ngram_alpha", 0)
        self.ngram_beta = decoding_params.get("ngram_beta", 0)
        self.ngram_offset = decoding_params.get("ngram_offset", 100)

        # LM
        self.lm = None
        self.lm_weight = decoding_params.get("lm_weight", 0)
        self.lm_tmp = decoding_params.get("lm_tmp", 1)

        # Distributed Computing
        self.is_distributed = False
        self.rank = 0
        self.is_parallel = False

        # Model Name
        self.name = name

    def compile(self, training_params):

        # Optimizers
        if training_params["optimizer"] == "Adam":

            # Adam
            self.optimizer = optim.Adam(
                params=self.parameters(), 
                lr=0, 
                betas=(training_params["beta1"], training_params["beta2"]), 
                eps=training_params["eps"], 
                weight_decay=training_params["weight_decay"])

        elif training_params["optimizer"] == "SGD":

            # SGD
            self.optimizer = optim.SGD(
                params=self.parameters, 
                lr=0, 
                momentum=training_params["momentum"], 
                weight_decay=training_params["weight_decay"])

        # LR Schedulers
        if training_params["lr_schedule"] == "Constant":
            
            # Constant LR
            self.scheduler = constant_learning_rate_scheduler(
                optimizer=self.optimizer,
                lr_value=training_params["lr_value"])

        elif training_params["lr_schedule"] == "ConstantWithDecay":
            
            # Constant With Decay LR
            self.scheduler = constant_with_decay_learning_rate_scheduler(
                optimizer=self.optimizer,
                lr_values=training_params["lr_values"],
                decay_steps=training_params["decay_steps"])

        elif training_params["lr_schedule"] == "Transformer":

            # Transformer LR
            self.scheduler = transformer_learning_rate_scheduler(
                optimizer=self.optimizer, 
                dim_model=training_params["schedule_dim"], 
                warmup_steps=training_params["warmup_steps"], 
                K=training_params["K"])

        elif training_params["lr_schedule"] == "ExpDecayTransformer":

            # Exp Decay Transformer LR
            self.scheduler = exponential_decay_transformer_learning_rate_scheduler(
                optimizer=self.optimizer, 
                warmup_steps=training_params["warmup_steps"], 
                lr_max=training_params["lr_max"] if training_params.get("lr_max", None) else training_params["K"] * training_params["schedule_dim"]**-0.5 * training_params["warmup_steps"]**-0.5, 
                alpha=training_params["alpha"], 
                end_step=training_params["end_step"])

        elif training_params["lr_schedule"] == "Cosine":

            # Cosine Annealing LR
            self.scheduler = cosine_annealing_learning_rate_scheduler(
                optimizer=self.optimizer, 
                warmup_steps=training_params["warmup_steps"], 
                lr_max=training_params["lr_max"] if training_params.get("lr_max", None) else training_params["K"] * training_params["schedule_dim"]**-0.5 * training_params["warmup_steps"]**-0.5, 
                lr_min= training_params["lr_min"], 
                end_step=training_params["end_step"])

        # Init LR
        self.scheduler.step()

    def num_params(self):

        return sum([p.numel() for p in self.parameters()])

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters :", self.num_params())
        if show_dict:
            for key, value in self.state_dict().items():
                print("{:<64} {:<16} mean {:<16.4f} std {:<16.4f}".format(key, str(tuple(value.size())), value.float().mean(), value.float().std()))

    def distribute_strategy(self, rank):

        self.rank = rank
        self.is_distributed = True

    def parallel_strategy(self):

        self.is_parallel = True

    def fit(self, dataset_train, epochs, dataset_val=None, val_steps=None, verbose_val=False, initial_epoch=0, callback_path=None, steps_per_epoch=None, mixed_precision=False, accumulated_steps=1, saving_period=1, val_period=1):

        # Model Device
        device = next(self.parameters()).device

        # Mixed Precision Gradient Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

        # Init Training
        acc_step = 0
        self.optimizer.zero_grad()

        # Callbacks
        if self.rank == 0 and callback_path is not None:

             # Create Callbacks
            if not os.path.isdir(callback_path):
                os.makedirs(callback_path)

            # Create Writer
            writer = SummaryWriter(callback_path + "logs")

        else:

            writer = None

        # Sample Synaptic Noise
        if self.vn_start_step is not None:
            if self.scheduler.model_step >= self.vn_start_step:
                self.decoder.apply(lambda m: sample_synaptic_noise(m, self.is_distributed))

        # Try Catch
        try:

            # Training Loop
            for epoch in range(initial_epoch, epochs):

                # Sync sampler if distributed
                if self.is_distributed:
                    dataset_train.sampler.set_epoch(epoch)

                # Epoch Init
                if self.rank == 0:
                    print("Epoch {}/{}".format(epoch + 1, epochs))
                    epoch_iterator = tqdm(dataset_train, total=steps_per_epoch * accumulated_steps if steps_per_epoch else None)
                else:
                    epoch_iterator = dataset_train
                epoch_loss = 0.0

                # Training Mode
                self.train()

                # Epoch training
                for step, batch in enumerate(epoch_iterator):

                    # Load batch to model device
                    batch = [elt.to(device) for elt in batch]

                    # Encoder Frozen Steps
                    if self.encoder_frozen_steps:
                        if self.scheduler.model_step > self.encoder_frozen_steps:
                            self.encoder.requires_grad_(True)
                        else:
                            self.encoder.requires_grad_(False)

                    # Automatic Mixed Precision Casting (model prediction + loss computing)
                    with torch.cuda.amp.autocast(enabled=mixed_precision):
                        pred = self.forward(batch)
                        loss_mini = self.criterion(batch, pred)
                        loss = loss_mini / accumulated_steps

                    # Accumulate gradients
                    scaler.scale(loss).backward()

                    # Update Epoch Variables
                    acc_step += 1
                    epoch_loss += loss_mini.detach()

                    # Continue Accumulating
                    if acc_step < accumulated_steps:
                        continue

                    # Update Parameters, Zero Gradients and Update Learning Rate
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    acc_step = 0

                    # Sample Synaptic Noise
                    if self.vn_start_step is not None:
                        if self.scheduler.model_step >= self.vn_start_step:
                            self.decoder.apply(lambda m: sample_synaptic_noise(m, self.is_distributed))

                    # Step Print
                    if self.rank == 0:
                        epoch_iterator.set_description("model step: {} - mean loss {:.4f} - batch loss: {:.4f} - learning rate: {:.6f}".format(self.scheduler.model_step, epoch_loss / (step + 1), loss_mini, self.optimizer.param_groups[0]['lr']))

                    # Logs Step
                    if self.rank == 0 and writer is not None and (step + 1) % 10 == 0:
                        writer.add_scalar('Training/Loss', loss_mini, self.scheduler.model_step)
                        writer.add_scalar('Training/LearningRate',  self.optimizer.param_groups[0]['lr'], self.scheduler.model_step)

                    # Step per Epoch
                    if steps_per_epoch is not None:
                        if step + 1 >= steps_per_epoch * accumulated_steps:
                            break

                # Reduce Epoch Loss among devices
                if self.is_distributed:
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(epoch_loss)
                    epoch_loss /= torch.distributed.get_world_size()

                # Logs Epoch
                if self.rank == 0 and writer is not None:
                    writer.add_scalar('Training/MeanLoss', epoch_loss / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else dataset_train.__len__()),  epoch + 1)

                # Validation
                if (epoch + 1) % val_period == 0:

                    # Validation Dataset
                    if dataset_val:

                        # Multiple Validation Datasets
                        if isinstance(dataset_val, dict):

                            for dataset_name, dataset in dataset_val.items():

                                # Evaluate
                                wer, truths, preds, val_loss = self.evaluate(dataset, val_steps, verbose_val, eval_loss=True)

                                # Print wer
                                if self.rank == 0:
                                    print("{} wer : {:.2f}% - loss : {:.4f}".format(dataset_name, 100 * wer, val_loss))

                                # Logs Validation
                                if self.rank == 0 and writer is not None:
                                    writer.add_scalar('Validation/WER/{}'.format(dataset_name), 100 * wer, epoch + 1)
                                    writer.add_scalar('Validation/MeanLoss/{}'.format(dataset_name), val_loss, epoch + 1)
                                    writer.add_text('Validation/Predictions/{}'.format(dataset_name), "GroundTruth : " + truths[0] + " / Prediction : " + preds[0], epoch + 1)

                        else:

                            # Evaluate
                            wer, truths, preds, val_loss = self.evaluate(dataset_val, val_steps, verbose_val, eval_loss=True)

                            # Print wer
                            if self.rank == 0:
                                print("Val wer : {:.2f}% - Val loss : {:.4f}".format(100 * wer, val_loss))

                            # Logs Validation
                            if self.rank == 0 and writer is not None:
                                writer.add_scalar('Validation/WER', 100 * wer, epoch + 1)
                                writer.add_scalar('Validation/MeanLoss', val_loss, epoch + 1)
                                writer.add_text('Validation/Predictions', "GroundTruth : " + truths[0] + " / Prediction : " + preds[0], epoch + 1)

                # Saving Checkpoint
                if (epoch + 1) % saving_period == 0:
                    if callback_path and self.rank == 0:
                        self.save(callback_path + "checkpoints_" + str(epoch + 1) + ".ckpt")

        # Exception Handler
        except Exception as e:

            if self.is_distributed:
                torch.distributed.destroy_process_group()

            if self.rank == 0 and writer is not None:
                writer.add_text('Exceptions', str(e))

            raise e

    def save(self, path, save_optimizer=True):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if save_optimizer else None,
            "model_step": self.scheduler.model_step,
            "tokenizer": self.tokenizer,
            "is_distributed": self.is_distributed or self.is_parallel
            }, path)

        # Print Model state
        if self.rank == 0:
            print("model saved at step {} / lr {:.6f}".format(self.scheduler.model_step, self.optimizer.param_groups[0]['lr']))

    def load(self, path):

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)

        # Model State Dict
        if checkpoint["is_distributed"] and not self.is_distributed:
            self.load_state_dict({key.replace(".module.", "."):value for key, value in checkpoint["model_state_dict"].items()})
        else:
            self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()})

        # Model Step
        self.scheduler.model_step = checkpoint["model_step"]

        # Optimizer State Dict
        if checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Tokenizer
        self.tokenizer = checkpoint["tokenizer"]

        # Print Model state
        if self.rank == 0:
            print("model loaded at step {} / lr {:.6f}".format(self.scheduler.model_step, self.optimizer.param_groups[0]['lr']))

    def evaluate(self, dataset_eval, eval_steps=None, verbose=False, beam_size=1, eval_loss=True):

        # Evaluzation Mode
        self.eval()

        # Model Device
        device = next(self.parameters()).device

        # Groundtruth / Prediction string lists
        speech_true = []
        speech_pred = []

        # Total wer / loss
        total_wer = 0.0
        total_loss = 0.0

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps)
        else: 
            eval_iterator = dataset_eval

        # Evaluation Loop
        for step, batch in enumerate(eval_iterator):

            batch = [elt.to(device) for elt in batch]

            # Sequence Prediction
            with torch.no_grad():

                if beam_size > 1:
                    outputs_pred = self.beam_search_decoding(batch[0], batch[2], beam_size)
                else:
                    outputs_pred = self.gready_search_decoding(batch[0], batch[2])

            # Sequence Truth
            outputs_true = self.tokenizer.decode(batch[1].tolist())

            # Compute Batch wer and Update total wer
            batch_wer = jiwer.wer(outputs_true, outputs_pred, standardize=True)
            total_wer += batch_wer

            # Update String lists
            speech_true += outputs_true
            speech_pred += outputs_pred

            # Prediction Verbose
            if verbose:
                print("Groundtruths :\n", outputs_true)
                print("Predictions :\n", outputs_pred)

            # Eval Loss
            if eval_loss:
                with torch.no_grad():
                    pred = self.forward(batch)
                    batch_loss = self.criterion(batch, pred)
                    total_loss += batch_loss

            # Step print
            if self.rank == 0:
                if eval_loss:
                    eval_iterator.set_description("mean batch wer {:.2f}% - batch wer: {:.2f}% - mean loss {:.4f} - batch loss: {:.4f}".format(100 * total_wer / (step + 1), 100 * batch_wer, total_loss / (step + 1), batch_loss))
                else:
                    eval_iterator.set_description("mean batch wer {:.2f}% - batch wer: {:.2f}%".format(100 * total_wer / (step + 1), 100 * batch_wer))

            # Evaluation Steps
            if eval_steps:
                if step + 1 >= eval_steps:
                    break

        # Reduce wer among devices
        if self.is_distributed:

            # Process Barrier
            torch.distributed.barrier()

            # All Gather Speech Truths and Predictions
            speech_true_gather = [None for _ in range(torch.distributed.get_world_size())]
            speech_pred_gather = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(speech_true_gather, speech_true)
            torch.distributed.all_gather_object(speech_pred_gather, speech_pred)
            speech_true = []
            speech_pred = []
            for truth in speech_true_gather:
                speech_true += truth
            for pred in speech_pred_gather:
                speech_pred += pred

            # All Reduce Total loss
            if eval_loss:
                torch.distributed.all_reduce(total_loss)
                total_loss /= torch.distributed.get_world_size()

        # Compute wer
        if total_wer / (eval_steps if eval_steps is not None else dataset_eval.__len__()) > 1:
            wer = 1
        else:
            wer = jiwer.wer(speech_true, speech_pred, standardize=True)

        # Compute loss
        if eval_loss:
            loss = total_loss / (eval_steps if eval_steps is not None else dataset_eval.__len__())

        # Return word error rate, groundtruths and predictions
        return wer, speech_true, speech_pred, loss if eval_loss else None

    def swa(self, dataset, callback_path, start_epoch, end_epoch, epochs_list=None, update_steps=None, swa_type="equal", swa_decay=0.9):

        # Model device
        device = next(self.parameters()).device

        # Create SWA Model
        if swa_type == "equal":
            swa_model = torch.optim.swa_utils.AveragedModel(self)
        elif swa_type == "exp":
            swa_model = torch.optim.swa_utils.AveragedModel(self, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: (1 - swa_decay) * averaged_model_parameter + swa_decay * model_parameter)

        if self.rank == 0:
            if epochs_list:
                print("Stochastic Weight Averaging on checkpoints : {}".format(epochs_list))
            else:
                print("Stochastic Weight Averaging on checkpoints : {}-{}".format(start_epoch, end_epoch))

        # Update SWA Model Params
        if epochs_list:

            for epoch in epochs_list:

                # Load Model Checkpoint
                self.load(callback_path + "checkpoints_" + str(epoch) + ".ckpt")

                # Update SWA Model
                swa_model.update_parameters(self)

        else:

            for epoch in range(int(start_epoch), int(end_epoch) + 1):

                # Load Model Checkpoint
                self.load(callback_path + "checkpoints_" + str(epoch) + ".ckpt")

                # Update SWA Model
                swa_model.update_parameters(self)

        # Load SWA Model Params
        self.load_state_dict({key[7:]:value for key, value in swa_model.state_dict().items() if key != "n_averaged"})

        if self.rank == 0:
            print("Updating Batch Normalization Statistics")

        # Init
        self.train()
        if self.rank == 0:
            dataset_iterator = tqdm(dataset, total=update_steps)
        else:
            dataset_iterator = dataset

        # Update Batch Normalization Statistics
        for step, batch in enumerate(dataset_iterator):

            # Load batch to model device
            batch = [elt.to(device) for elt in batch]

            # Forward Encoder
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    self.encoder.forward(batch[0], batch[2])

            # update_steps
            if update_steps is not None:
                if step + 1 == update_steps:
                    break

        # Save Model
        if self.rank == 0:
            if epochs_list:
                self.save(callback_path + "checkpoints_swa-" + swa_type + "-" + "list" + "-" + epochs_list[0] + "-"  + epochs_list[-1] + ".ckpt", save_optimizer=False)
            else:
                self.save(callback_path + "checkpoints_swa-" + swa_type + "-" + start_epoch + "-"  + end_epoch + ".ckpt", save_optimizer=False)

        # Barrier
        if self.is_distributed:
            torch.distributed.barrier()

    def eval_time(self, dataset_eval, eval_steps=None, beam_size=1, rnnt_max_consec_dec_steps=None, profiler=False):

        def decode():

            # Start Timer
            start = time.time()

            # Evaluation Loop
            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                # Sequence Prediction
                with torch.no_grad():

                    if beam_size > 1:
                        outputs_pred = self.beam_search_decoding(batch[0], batch[2], beam_size)
                    else:
                        if rnnt_max_consec_dec_steps is not None:
                            outputs_pred = self.gready_search_decoding(batch[0], batch[2], rnnt_max_consec_dec_steps)
                        else:
                            outputs_pred = self.gready_search_decoding(batch[0], batch[2])

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break
            # Stop Timer
            return time.time() - start

        # Model Device
        device = next(self.parameters()).device

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps)
        else: 
            eval_iterator = dataset_eval

        # Decoding
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = decode()
        else:
            timer = decode()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer

    def eval_time_encoder(self, dataset_eval, eval_steps=None, profiler=False):

        def forward():

            # Start Timer
            start = time.time()

            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                with torch.no_grad():
                    x, x_len, att = self.encoder.forward(batch[0], batch[2])

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break

            # Stop Timer
            return time.time() - start

        # Model Device
        device = next(self.parameters()).device

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps)
        else: 
            eval_iterator = dataset_eval

        # Forward
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = forward()
        else:
            timer = forward()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer

    def eval_time_decoder(self, dataset_eval, eval_steps=None, profiler=False):

        def forward():

            # Start Timer
            start = time.time()

            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                hidden = None

                for i in range(batch[1].size(1)):
                    with torch.no_grad():
                        _, hidden = self.decoder.forward(batch[1][:, i:i+1], hidden)

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break

            # Stop Timer
            return time.time() - start

        # Model Device
        device = next(self.parameters()).device

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps)
        else: 
            eval_iterator = dataset_eval

        # Forward
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = forward()
        else:
            timer = forward()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer


