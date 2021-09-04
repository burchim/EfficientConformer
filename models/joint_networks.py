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

# Layers 
from models.layers import (
    Linear
)

# Activations Functions
from models.activations import (
    Swish
)

###############################################################################
# Joint Networks
###############################################################################

class JointNetwork(nn.Module):

    def __init__(self, dim_encoder, dim_decoder, vocab_size, params):
        super(JointNetwork, self).__init__()

        assert params["act"] in ["tanh", "relu", "swish", None]
        assert params["joint_mode"] in ["concat", "sum"]

        # Model layers
        if params["dim_model"] is not None:

            # Linear Layers
            self.linear_encoder = Linear(dim_encoder, params["dim_model"])
            self.linear_decoder = Linear(dim_decoder, params["dim_model"])

            # Joint Mode
            if params["joint_mode"] == "concat":
                self.joint_mode = "concat"
                self.linear_joint = Linear(2 * params["dim_model"], vocab_size)
            elif params["joint_mode"] == "sum":
                self.joint_mode = 'sum'
                self.linear_joint = Linear(params["dim_model"], vocab_size)
        else:

            # Linear Layers
            self.linear_encoder = nn.Identity()
            self.linear_decoder = nn.Identity()

            # Joint Mode
            if params["joint_mode"] == "concat":
                self.joint_mode = "concat"
                self.linear_joint = Linear(dim_encoder + dim_decoder, vocab_size)
            elif params["joint_mode"] == "sum":
                assert dim_encoder == dim_decoder
                self.joint_mode = 'sum'
                self.linear_joint = Linear(dim_encoder, vocab_size)

        # Model Act Function
        if params["act"] == "tanh":
            self.act = nn.Tanh()
        elif params["act"] == "relu":
            self.act = nn.ReLU()
        elif params["act"] == "swish":
            self.act = Swish()
        else:
            self.act = nn.Identity()

    def forward(self, f, g):

        f = self.linear_encoder(f)
        g = self.linear_decoder(g)

        # Training or Eval Loss
        if self.training or (len(f.size()) == 3 and len(g.size()) == 3):
            f = f.unsqueeze(2) # (B, T, 1, D)
            g = g.unsqueeze(1) # (B, 1, U + 1, D)

            f = f.repeat([1, 1, g.size(2), 1]) # (B, T, U + 1, D)
            g = g.repeat([1, f.size(1), 1, 1]) # (B, T, U + 1, D)

        # Joint Encoder and Decoder
        if self.joint_mode == "concat":
            joint = torch.cat([f, g], dim=-1) # Training : (B, T, U + 1, 2D) / Decoding : (B, 2D)
        elif self.joint_mode == "sum":
            joint = f + g # Training : (B, T, U + 1, D) / Decoding : (B, D)

        # Act Function
        joint = self.act(joint)

        # Output Linear Projection
        outputs = self.linear_joint(joint) # Training : (B, T, U + 1, V) / Decoding : (B, V)
        
        return outputs