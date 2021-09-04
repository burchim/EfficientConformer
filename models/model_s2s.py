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

# Base Model
from models.model import Model

# Encoders
from models.encoders import (
    ConformerEncoder
)

# Decoders
from models.decoders import (
    ConformerCrossDecoder,
    TransformerCrossDecoder
)

# Losses
from models.losses import (
    LossCE
)

# Ngram
import kenlm

class ModelS2S(Model):

    def __init__(self, encoder_params, decoder_params, tokenizer_params, training_params, decoding_params, name):
        super(ModelS2S, self).__init__(tokenizer_params, training_params, decoding_params, name)

        # Not Implemented
        raise Exception("Sequence-to-sequence model not implemented")

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoder(encoder_params)
        else:
            raise Exception("Unknown encoder architecture:", encoder_params["arch"])

        # Decoder
        if decoder_params["arch"] == "Conformer":
            self.decoder = ConformerCrossDecoder(decoder_params)
        elif decoder_params["arch"] == "Transformer":
            self.decoder = TransformerCrossDecoder(decoder_params)
        else:
            raise Exception("Unknown decoder architecture:", decoder_params["arch"])

        # Joint Network
        self.fc = nn.Linear(encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else encoder_params["dim_model"], tokenizer_params["vocab_size"])

        # Criterion
        self.criterion = LossCE()

        # Compile
        self.compile(training_params)

    def forward(self, batch):

        # Unpack Batch
        x, y, _ = batch

        # Audio Encoder (B, Taud) -> (B, T, Denc)
        x, _, attentions = self.encoder(x, None)

        # Add blank token
        y = torch.nn.functional.pad(y, pad=(1, 0, 0, 0), value=0)

        # Text Decoder (B, U + 1) -> (B, U + 1, Ddec)
        y = self.decoder(x, y)

        # FC Layer (B, T, Ddec) -> (B, T, V)
        logits = self.fc(y)

        return logits, attentions

    def distribute_strategy(self, rank):
        super(ModelS2S, self).distribute_strategy(rank)

        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.rank])
        self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder, device_ids=[self.rank])
        self.fc = torch.nn.parallel.DistributedDataParallel(self.fc, device_ids=[self.rank])

    def parallel_strategy(self):
        super(ModelS2S, self).parallel_strategy()

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)
        self.fc = torch.nn.DataParallel(self.fc)

    def summary(self, show_dict=False):

        print(self.name)
        print("Model Parameters :", self.num_params() - self.lm.num_params() if self.lm else self.num_params())
        print(" - Encoder Parameters :", sum([p.numel() for p in self.encoder.parameters()]))
        print(" - Decoder Parameters :", sum([p.numel() for p in self.decoder.parameters()]))
        print(" - Joint Parameters :", sum([p.numel() for p in self.joint_network.parameters()]))

        if self.lm:
            print("LM Parameters :", self.lm.num_params())

        if show_dict:
            for key, value in self.state_dict().items():
                print("{:<64} {:<16} mean {:<16.4f} std {:<16.4f}".format(key, str(tuple(value.size())), value.float().mean(), value.float().std()))
