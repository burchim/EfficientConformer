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

# Modules
from models.modules import (
    FeedForwardModule,
    MultiHeadSelfAttentionModule,
    ConvolutionModule
)

# Layers
from models.layers import (
    Conv1d,
    Transpose
)

class ConformerBlock(nn.Module):

    def __init__(
        self, 
        dim_model, 
        dim_expand, 
        ff_ratio, 
        num_heads, 
        kernel_size, 
        att_group_size, 
        att_kernel_size,
        linear_att,
        Pdrop, 
        relative_pos_enc, 
        max_pos_encoding, 
        conv_stride,
        att_stride,
        causal
    ):
        super(ConformerBlock, self).__init__()

        # Feed Forward Module 1
        self.feed_forward_module1 = FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio,
            Pdrop=Pdrop, 
            act="swish",
            inner_dropout=True
        )

        # Multi-Head Self-Attention Module
        self.multi_head_self_attention_module = MultiHeadSelfAttentionModule(
            dim_model=dim_model, 
            num_heads=num_heads,  
            Pdrop=Pdrop, 
            max_pos_encoding=max_pos_encoding,
            relative_pos_enc=relative_pos_enc, 
            causal=causal,
            group_size=att_group_size,
            kernel_size=att_kernel_size,
            stride=att_stride,
            linear_att=linear_att
        )

        # Convolution Module
        self.convolution_module = ConvolutionModule(
            dim_model=dim_model,
            dim_expand=dim_expand,
            kernel_size=kernel_size, 
            Pdrop=Pdrop, 
            stride=conv_stride,
            padding="causal" if causal else "same"
        )

        # Feed Forward Module 2
        self.feed_forward_module2 = FeedForwardModule(
            dim_model=dim_expand, 
            dim_ffn=dim_expand * ff_ratio,
            Pdrop=Pdrop, 
            act="swish",
            inner_dropout=True
        )

        # Block Norm
        self.norm = nn.LayerNorm(dim_expand, eps=1e-6)

        # Attention Residual
        self.att_res = nn.Sequential(
            Transpose(1, 2),
            nn.MaxPool1d(kernel_size=1, stride=att_stride),
            Transpose(1, 2)
        ) if att_stride > 1 else nn.Identity()

        # Convolution Residual
        self.conv_res = nn.Sequential(
            Transpose(1, 2),
            Conv1d(dim_model, dim_expand, kernel_size=1, stride=conv_stride),
            Transpose(1, 2)
        ) if dim_model != dim_expand else nn.Sequential(
            Transpose(1, 2),
            nn.MaxPool1d(kernel_size=1, stride=conv_stride),
            Transpose(1, 2)
        ) if conv_stride > 1 else nn.Identity()

        # Bloc Stride
        self.stride = conv_stride * att_stride

    def forward(self, x, mask=None, hidden=None):

        # FFN Module 1
        x = x + 1/2 * self.feed_forward_module1(x)

        # MHSA Module
        x_att, attention, hidden = self.multi_head_self_attention_module(x, mask, hidden)
        x = self.att_res(x) + x_att

        # Conv Module
        x = self.conv_res(x) + self.convolution_module(x)

        # FFN Module 2
        x = x + 1/2 * self.feed_forward_module2(x)

        # Block Norm
        x = self.norm(x)

        return x, attention, hidden

class TransformerBlock(nn.Module):

    def __init__(self, dim_model, ff_ratio, num_heads, Pdrop, max_pos_encoding, relative_pos_enc, causal):
        super(TransformerBlock, self).__init__()

        # Muti-Head Self-Attention Module
        self.multi_head_self_attention_module = MultiHeadSelfAttentionModule(
            dim_model=dim_model,
            num_heads=num_heads,
            Pdrop=Pdrop,
            max_pos_encoding=max_pos_encoding,
            relative_pos_enc=relative_pos_enc,
            causal=causal,
            group_size=1,
            kernel_size=1,
            stride=1,
            efficient_att=False
        )

        # Feed Forward Module
        self.feed_forward_module = FeedForwardModule(
            dim_model=dim_model, 
            dim_ffn=dim_model * ff_ratio, 
            Pdrop=Pdrop, 
            act="relu",
            inner_dropout=False
        )

    def forward(self, x, mask=None, hidden=None):

        # Muti-Head Self-Attention Module
        x_att, attention, hidden = self.multi_head_self_attention_module(x, mask, hidden)
        x = x + x_att

        # Feed Forward Module
        x = x + self.feed_forward_module(x)

        return x, attention, hidden