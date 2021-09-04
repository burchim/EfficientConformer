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

# Decoder
from models.decoders import (
    RnnDecoder,
    TransformerDecoder
)

# Losses
from models.losses import (
    LossCE
)
  
class LanguageModel(Model):

    def __init__(self, lm_params, tokenizer_params, training_params, decoding_params, name):
        super(LanguageModel, self).__init__(tokenizer_params, training_params, decoding_params, name)

        # Language Model
        if lm_params["arch"] == "RNN":
            self.decoder = RnnDecoder(lm_params)
        elif lm_params["arch"] == "Transformer":
            self.decoder = TransformerDecoder(lm_params)
        else:
            raise Exception("Unknown model architecture:", lm_params["arch"])

        # FC Layer
        self.fc = nn.Linear(lm_params["dim_model"], tokenizer_params["vocab_size"])

        # Criterion
        self.criterion = LossCE()

        # Compile
        self.compile(training_params)

    def decode(self, x, hidden):

        # Text Decoder (1, 1) -> (1, 1, Dlm)
        logits, hidden = self.decoder(x, hidden)

        # FC Layer (1, 1, Dlm) -> (1, 1, V)
        logits = self.fc(logits)

        return logits, hidden

    def forward(self, batch):

        # Unpack Batch
        x, x_len, y = batch

        # Add blank token
        x = torch.nn.functional.pad(x, pad=(1, 0, 0, 0), value=0)
        if x_len is not None:
            x_len = x_len + 1

        # Text Decoder (B, U + 1) -> (B, U + 1, Dlm)
        logits, _ = self.decoder(x, None, x_len)

        # FC Layer (B, U + 1, Dlm) -> (B, U + 1, V)
        logits = self.fc(logits)

        return logits

    def gready_search_decoding(self, x, x_len):
        
        return [""]