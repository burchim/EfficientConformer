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

# Positional Encodings and Masks
from models.attentions import (
    SinusoidalPositionalEncoding,
    StreamingMask
)

# Blocks
from models.blocks import (
    TransformerBlock,
    ConformerBlock
)

# Layers
from models.layers import (
    Embedding,
    LSTM
)

###############################################################################
# Decoder Models
###############################################################################

class RnnDecoder(nn.Module):

    def __init__(self, params):
        super(RnnDecoder, self).__init__()

        self.embedding = Embedding(params["vocab_size"], params["dim_model"], padding_idx=0)
        self.rnn = LSTM(input_size=params["dim_model"], hidden_size=params["dim_model"], num_layers=params["num_layers"], batch_first=True, bidirectional=False)

    def forward(self, y, hidden, y_len=None):

        # Sequence Embedding (B, U + 1) -> (N, U + 1, D)
        y = self.embedding(y)

        # Pack padded batch sequences
        if y_len is not None:
            y = nn.utils.rnn.pack_padded_sequence(y, y_len.cpu(), batch_first=True, enforce_sorted=False)

        # Hidden state provided
        if hidden is not None:
            y, hidden = self.rnn(y, hidden)
        # None hidden state
        else:
            y, hidden = self.rnn(y)

        # Pad packed batch sequences
        if y_len is not None:
            y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)

        # return last layer steps outputs and every layer last step hidden state
        return y, hidden

class TransformerDecoder(nn.Module):

    def __init__(self, params):
        super(TransformerDecoder, self).__init__()

        # Look Ahead Mask
        self.look_ahead_mask = StreamingMask(left_context=params.get("left_context", params["max_pos_encoding"]), right_context=0)

        # Embedding Layer
        self.embedding = nn.Embedding(params["vocab_size"], params["dim_model"], padding_idx=0)

        # Dropout
        self.dropout = nn.Dropout(p=params["Pdrop"])

        # Sinusoidal Positional Encodings
        self.pos_enc = None if params["relative_pos_enc"] else SinusoidalPositionalEncoding(params["max_pos_encoding"], params["dim_model"])

        # Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(
            dim_model=params["dim_model"], 
            ff_ratio=params["ff_ratio"], 
            num_heads=params["num_heads"], 
            Pdrop=params["Pdrop"], 
            max_pos_encoding=params["max_pos_encoding"],
            relative_pos_enc=params["relative_pos_enc"],
            causal=True
        ) for block_id in range(params["num_blocks"])])

    def forward(self, y, hidden, y_len=None):

        # Look Ahead Mask
        if hidden == None:
            mask = self.look_ahead_mask(y, y_len)
        else:
            mask = None

        # Linear Proj
        y = self.embedding(y)

        # Dropout
        y = self.dropout(y)

        # Sinusoidal Positional Encodings
        if self.pos_enc is not None:
            y = y + self.pos_enc(y.size(0), y.size(1))

        # Transformer Blocks
        attentions = []
        hidden_new = []
        for block_id, block in enumerate(self.blocks):

            # Hidden State Provided
            if hidden is not None:
                y, attention, block_hidden = block(y, mask, hidden[block_id])
            else:
                y, attention, block_hidden = block(y, mask)

            # Update Hidden / Att Maps
            if not self.training:
                attentions.append(attention)
                hidden_new.append(block_hidden)

        return y, hidden_new

class ConformerDecoder(nn.Module):

    def __init__(self, params):
        super(ConformerDecoder, self).__init__()

        # Look Ahead Mask
        self.look_ahead_mask = StreamingMask(left_context=params.get("left_context", params["max_pos_encoding"]), right_context=0)

        # Embedding Layer
        self.embedding = nn.Embedding(params["vocab_size"], params["dim"], padding_idx=0)

        # Dropout
        self.dropout = nn.Dropout(p=params["Pdrop"])

        # Sinusoidal Positional Encodings
        self.pos_enc = None if params["relative_pos_enc"] else SinusoidalPositionalEncoding(params["max_pos_encoding"], params["dim_model"])

        # Conformer Layers
        self.blocks = nn.ModuleList([ConformerBlock(
            dim_model=params["dim_model"],
            dim_expand=params["dim_model"],
            ff_ratio=params["ff_ratio"],
            num_heads=params["num_heads"], 
            kernel_size=params["kernel_size"],
            att_group_size=1,
            att_kernel_size=None,
            Pdrop=params["Pdrop"], 
            relative_pos_enc=params["relative_pos_enc"], 
            max_pos_encoding=params["max_pos_encoding"],
            conv_stride=1,
            att_stride=1,
            causal=True
        ) for block_id in range(params["num_blocks"])])

    def forward(self, y, hidden, y_len=None):

        # Hidden state provided
        if hidden is not None:
            y = torch.cat([hidden, y], axis=1)

        # Look Ahead Mask
        mask = self.look_ahead_mask(y, y_len)

        # Update Hidden
        hidden_new = y

        # Linear Proj
        y = self.embedding(y)

        # Dropout
        y = self.dropout(y)

        # Sinusoidal Positional Encodings
        if self.pos_enc is not None:
            y = y + self.pos_enc(y.size(0), y.size(1))

        # Transformer Blocks
        attentions = []
        for block in self.blocks:
            y, attention = block(y, mask)
            attentions.append(attention)

        if hidden is not None:
            y = y[:, -1:]

        return y, hidden_new