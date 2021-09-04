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
import torch.nn.functional as F
import torch._VF as _VF
from torch.nn.modules.utils import _single, _pair

# Activation Functions
from models.activations import (
    Swish
)

###############################################################################
# Layers
###############################################################################

class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise
            
        # Apply Weight
        return F.linear(input, weight, self.bias)

class Conv1d(nn.Conv1d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride = 1, 
        padding = "same", 
        dilation = 1, 
        groups = 1, 
        bias = True
    ):
        super(Conv1d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode="zeros")

        # Assert
        assert padding in ["valid", "same", "causal"]

        # Padding
        if padding == "valid":
            self.pre_padding = None
        elif padding == "same":
            self.pre_padding = nn.ConstantPad1d(padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2), value=0)
        elif padding == "causal":
            self.pre_padding = nn.ConstantPad1d(padding=(kernel_size - 1, 0), value=0)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise

        # Padding
        if self.pre_padding is not None:
            input = self.pre_padding(input)

        # Apply Weight
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros'):
        super(Conv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise

        # Apply Weight
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LSTM(nn.LSTM):

    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional):
        super(LSTM, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=batch_first, 
            bidirectional=bidirectional)

        # Variational Noise
        self.noises = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noises = []
        for i in range(0, len(self._flat_weights), 4):
            self.noises.append(torch.normal(mean=0.0, std=1.0, size=self._flat_weights[i].size(), device=self._flat_weights[i].device, dtype=self._flat_weights[i].dtype))
            self.noises.append(torch.normal(mean=0.0, std=1.0, size=self._flat_weights[i+1].size(), device=self._flat_weights[i+1].device, dtype=self._flat_weights[i+1].dtype))

        # Broadcast Noise
        if distributed:
            for noise in self.noises:
                torch.distributed.broadcast(noise, 0)

    def forward(self, input, hx=None):  # noqa: F811

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        # Add Noise
        if self.noises is not None and self.training:
            weight = []
            for i in range(0, len(self.noises), 2):
                weight.append(self._flat_weights[2*i] + self.vn_std * self.noises[i])
                weight.append(self._flat_weights[2*i+1] + self.vn_std * self.noises[i+1])
                weight.append(self._flat_weights[2*i+2])
                weight.append(self._flat_weights[2*i+3])
        else:
            weight = self._flat_weights

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, weight, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, weight, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            output_packed = nn.utils.rnn.PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


class Embedding(nn.Embedding): 

    def __init__(self, num_embeddings, embedding_dim, padding_idx = None):
        super(Embedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise

        # Apply Weight
        return F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

class IdentityProjection(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(IdentityProjection, self).__init__()

        assert output_dim > input_dim
        self.linear = Linear(input_dim, output_dim - input_dim)

    def forward(self, x):

        # (B, T, Dout - Din)
        proj = self.linear(x)

        # (B, T, Dout)
        x = torch.cat([x, proj], dim=-1)

        return x

class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv1d, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            Conv1d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, stride=stride),
            Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            Swish()
        )

    def forward(self, x):
        return self.layers(x)

class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)