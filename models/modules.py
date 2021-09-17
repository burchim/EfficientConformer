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
import torchaudio

# Attentions
from models.attentions import (
    # Abs Attentions
    MultiHeadAttention,
    GroupedMultiHeadAttention,
    LocalMultiHeadAttention,
    StridedMultiHeadAttention,
    StridedLocalMultiHeadAttention,
    MultiHeadLinearAttention,
    # Rel Attentions
    RelPosMultiHeadSelfAttention,
    GroupedRelPosMultiHeadSelfAttention,
    LocalRelPosMultiHeadSelfAttention,
    StridedRelPosMultiHeadSelfAttention,
    StridedLocalRelPosMultiHeadSelfAttention
)

# Layers
from models.layers import (
    Linear,
    Conv1d,
    Transpose,
    DepthwiseSeparableConv1d
)

# Activations
from models.activations import (
    Swish,
    Glu
)

###############################################################################
# Audio Preprocessing
###############################################################################

class AudioPreprocessing(nn.Module):

    """Audio Preprocessing

    Computes mel-scale log filter banks spectrogram

    Args:
        sample_rate: Audio sample rate
        n_fft: FFT frame size, creates n_fft // 2 + 1 frequency bins.
        win_length_ms: FFT window length in ms, must be <= n_fft
        hop_length_ms: length of hop between FFT windows in ms
        n_mels: number of mel filter banks
        normalize: whether to normalize mel spectrograms outputs
        mean: training mean
        std: training std

    Shape:
        Input: (batch_size, audio_len)
        Output: (batch_size, n_mels, audio_len // hop_length + 1)
    
    """

    def __init__(self, sample_rate, n_fft, win_length_ms, hop_length_ms, n_mels, normalize, mean, std):
        super(AudioPreprocessing, self).__init__()
        self.win_length = int(sample_rate * win_length_ms) // 1000
        self.hop_length = int(sample_rate * hop_length_ms) // 1000
        self.Spectrogram = torchaudio.transforms.Spectrogram(n_fft, self.win_length, self.hop_length)
        self.MelScale = torchaudio.transforms.MelScale(n_mels, sample_rate, f_min=0, f_max=8000, n_stft=n_fft // 2 + 1)
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, x, x_len):

        # Short Time Fourier Transform (B, T) -> (B, n_fft // 2 + 1, T // hop_length + 1)
        x = self.Spectrogram(x)

        # Mel Scale (B, n_fft // 2 + 1, T // hop_length + 1) -> (B, n_mels, T // hop_length + 1)
        x = self.MelScale(x)
        
        # Energy log, autocast disabled to prevent float16 overflow
        x = (x.float() + 1e-9).log().type(x.dtype)

        # Compute Sequence lengths 
        if x_len is not None:
            x_len = x_len // self.hop_length + 1

        # Normalize
        if self.normalize:
            x = (x - self.mean) / self.std

        return x, x_len

class SpecAugment(nn.Module):

    """Spectrogram Augmentation

    Args:
        spec_augment: whether to apply spec augment
        mF: number of frequency masks
        F: maximum frequency mask size
        mT: number of time masks
        pS: adaptive maximum time mask size in %

    References:
        SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition, Park et al.
        https://arxiv.org/abs/1904.08779

        SpecAugment on Large Scale Datasets, Park et al.
        https://arxiv.org/abs/1912.05533

    """

    def __init__(self, spec_augment, mF, F, mT, pS):
        super(SpecAugment, self).__init__()
        self.spec_augment = spec_augment
        self.mF = mF
        self.F = F
        self.mT = mT
        self.pS = pS

    def forward(self, x, x_len):

        # Spec Augment
        if self.spec_augment:
        
            # Frequency Masking
            for _ in range(self.mF):
                x = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False).forward(x)

            # Time Masking
            for b in range(x.size(0)):
                T = int(self.pS * x_len[b])
                for _ in range(self.mT):
                    x[b, :, :x_len[b]] = torchaudio.transforms.TimeMasking(time_mask_param=T).forward(x[b, :, :x_len[b]])

        return x

###############################################################################
# Conv Subsampling Modules
###############################################################################

class Conv1dSubsampling(nn.Module):

    """Conv1d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        in_dim: input feature dimension
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, in_dim, filters, kernel_size, norm, act):
        super(Conv1dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        # Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_dim if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, stride=2, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = (x_len - 1) // 2 + 1

        return x, x_len

class Conv2dSubsampling(nn.Module):

    """Conv2d Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act):
        super(Conv2dSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        # Conv 2D Subsampling Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, stride=2, padding=(kernel_size - 1) // 2), 
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = (x_len - 1) // 2 + 1

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)

        return x, x_len

class Conv2dPoolSubsampling(nn.Module):

    """Conv2d with Max Pooling Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act):
        super(Conv2dPoolSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        # Layers
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity()
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Layers
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = (x_len - 1) // 2 + 1

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)

        return x, x_len

class VGGSubsampling(nn.Module):

    """VGG style Subsampling Block

    Args:
        num_layers: number of strided convolution layers
        filters: list of convolution layers filters
        kernel_size: convolution kernel size
        norm: normalization
        act: activation function

    Shape:
        Input: (batch_size, in_dim, in_length)
        Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, num_layers, filters, kernel_size, norm, act):
        super(VGGSubsampling, self).__init__()

        # Assert
        assert norm in ["batch", "layer", "none"]
        assert act in ["relu", "swish", "none"]

        self.layers = nn.ModuleList([nn.Sequential(
            # Conv 1
            nn.Conv2d(1 if layer_id == 0 else filters[layer_id - 1], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity(),
            # Conv 2
            nn.Conv2d(filters[layer_id], filters[layer_id], kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(filters[layer_id]) if norm == "batch" else nn.LayerNorm(filters[layer_id]) if norm == "layer" else nn.Identity(),
            nn.ReLU() if act == "relu" else Swish() if act == "swish" else nn.Identity(),
            # Pooling
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) 
        ) for layer_id in range(num_layers)])

    def forward(self, x, x_len):

        # (B, D, T) -> (B, 1, D, T)
        x = x.unsqueeze(dim=1)

        # Stages
        for layer in self.layers:
            x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = x_len // 2

        # (B, C, D // S, T // S) -> (B,  C * D // S, T // S)
        batch_size, channels, subsampled_dim, subsampled_length = x.size()
        x = x.reshape(batch_size, channels * subsampled_dim, subsampled_length)
        
        return x, x_len

###############################################################################
# Conformer Modules
###############################################################################

class FeedForwardModule(nn.Module):

    """Transformer Feed Forward Module

    Args:
        dim_model: model feature dimension
        dim_ffn: expanded feature dimension
        Pdrop: dropout probability
        act: inner activation function
        inner_dropout: whether to apply dropout after the inner activation function

    Input: (batch size, length, dim_model)
    Output: (batch size, length, dim_model)
    
    """

    def __init__(self, dim_model, dim_ffn, Pdrop, act, inner_dropout):
        super(FeedForwardModule, self).__init__()

        # Assert
        assert act in ["relu", "swish"]

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model, eps=1e-6),
            Linear(dim_model, dim_ffn),
            Swish() if act=="swish" else nn.ReLU(),
            nn.Dropout(p=Pdrop) if inner_dropout else nn.Identity(),
            Linear(dim_ffn, dim_model),
            nn.Dropout(p=Pdrop)
        )

    def forward(self, x):
        return self.layers(x)

class MultiHeadSelfAttentionModule(nn.Module):

    """Multi-Head Self-Attention Module

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        Pdrop: residual dropout probability
        max_pos_encoding: maximum position
        relative_pos_enc: whether to use relative postion embedding
        causal: True for causal attention with masked future context
        group_size: Attention group size
        kernel_size: Attention kernel size
        stride: Query stride
        linear_att: whether to use multi-head linear self-attention

    """

    def __init__(self, dim_model, num_heads, Pdrop, max_pos_encoding, relative_pos_enc, causal, group_size, kernel_size, stride, linear_att):
        super(MultiHeadSelfAttentionModule, self).__init__()

        # Assert
        assert not (group_size > 1 and kernel_size is not None), "Local grouped attention not implemented"
        assert not (group_size > 1 and stride > 1 is not None), "Strided grouped attention not implemented"
        assert not (linear_att and relative_pos_enc), "Linear attention requires absolute positional encodings"

        # Pre Norm
        self.norm = nn.LayerNorm(dim_model, eps=1e-6)

        # Multi-Head Linear Attention
        if linear_att:
            self.mhsa = MultiHeadLinearAttention(dim_model, num_heads)

        # Grouped Multi-Head Self-Attention
        elif group_size > 1:
            if relative_pos_enc:
                self.mhsa = GroupedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, group_size)
            else:
                self.mhsa = GroupedMultiHeadAttention(dim_model, num_heads, group_size)
        
        # Local Multi-Head Self-Attention
        elif kernel_size is not None and stride == 1:
            if relative_pos_enc:
                self.mhsa = LocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size)
            else:
                self.mhsa = LocalMultiHeadAttention(dim_model, num_heads, kernel_size)

        # Strided Multi-Head Self-Attention
        elif kernel_size is None and stride > 1:
            if relative_pos_enc:
                self.mhsa = StridedRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding, stride)
            else:
                self.mhsa = StridedMultiHeadAttention(dim_model, num_heads, stride)

        # Strided Local Multi-Head Self-Attention
        elif stride > 1 and kernel_size is not None:
            if relative_pos_enc:
                self.mhsa = StridedLocalRelPosMultiHeadSelfAttention(dim_model, num_heads, causal, kernel_size, stride)
            else:
                self.mhsa = StridedLocalMultiHeadAttention(dim_model, num_heads, kernel_size, stride)

        # Multi-Head Self-Attention
        else:
            if relative_pos_enc:
                self.mhsa = RelPosMultiHeadSelfAttention(dim_model, num_heads, causal, max_pos_encoding)
            else:
                self.mhsa = MultiHeadAttention(dim_model, num_heads)
            
        # Dropout
        self.dropout = nn.Dropout(Pdrop)

        # Module Params
        self.rel_pos_enc = relative_pos_enc
        self.linear_att = linear_att

    def forward(self, x, mask=None, hidden=None):

        # Pre Norm
        x = self.norm(x)

        # Multi-Head Self-Attention
        if self.linear_att:
            x, attention = self.mhsa(x, x, x)
        elif self.rel_pos_enc:
            x, attention, hidden = self.mhsa(x, x, x, mask, hidden)
        else:
            x, attention = self.mhsa(x, x, x, mask)

        # Dropout
        x = self.dropout(x)

        return x, attention, hidden

class ConvolutionModule(nn.Module):

    """Conformer Convolution Module

    Args:
        dim_model: input feature dimension
        dim_expand: output feature dimension
        kernel_size: 1D depthwise convolution kernel size
        Pdrop: residual dropout probability
        stride: 1D depthwise convolution stride
        padding: "valid", "same" or "causal"

    Input: (batch size, input length, dim_model)
    Output: (batch size, output length, dim_expand)
    
    """

    def __init__(self, dim_model, dim_expand, kernel_size, Pdrop, stride, padding):
        super(ConvolutionModule, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model, eps=1e-6),
            Transpose(1, 2),
            Conv1d(dim_model, 2 * dim_expand, kernel_size=1),
            Glu(dim=1),
            Conv1d(dim_expand, dim_expand, kernel_size, stride=stride, padding=padding, groups=dim_expand),
            nn.BatchNorm1d(dim_expand),
            Swish(),
            Conv1d(dim_expand, dim_expand, kernel_size=1),
            Transpose(1, 2),
            nn.Dropout(p=Pdrop)
        )

    def forward(self, x):
        return self.layers(x)

###############################################################################
# ContextNet Modules
###############################################################################

class ContextNetBlock(nn.Module):

    def __init__(self, num_layers, dim_in, dim_out, kernel_size, stride, causal, se_ratio, residual, padding):
        super(ContextNetBlock, self).__init__()

        # Conv Layers
        self.conv_layers = nn.Sequential(*[
            DepthwiseSeparableConv1d(dim_in if layer_id == 0 else dim_out, dim_out, kernel_size, stride if layer_id == num_layers - 1 else 1, causal) 
        for layer_id in range(num_layers)])

        # SE Module
        self.se_module = SqueezeAndExcitationModule(dim_out, se_ratio, "swish") if se_ratio is not None else None

        # Residual
        self.residual = nn.Sequential(
            Conv1d(dim_in, dim_out, kernel_size=1, stride=stride, groups=1, padding=padding),
            nn.BatchNorm1d(dim_out)
        ) if residual else None

        # Block Act
        self.act = Swish()

    def forward(self, x):

        # Conv Layers
        y = self.conv_layers(x)

        # SE Module
        if self.se_module is not None:
            y = self.se_module(y)

        # Residual
        if self.residual is not None:
            y = self.act(y + self.residual(x))

        return y  

class ContextNetSubsampling(nn.Module):

    def __init__(self, n_mels, dim_model, kernel_size, causal):
        super(ContextNetSubsampling, self).__init__()

        # Blocks
        self.blocks = nn.Sequential(*[ContextNetBlock(
            num_layers=1 if block_id == 0 else 5, 
            dim_in=n_mels if block_id == 0 else dim_model, 
            dim_out=dim_model, 
            kernel_size=kernel_size, 
            stride=2 if block_id in [3, 7] else 1, 
            causal=causal, 
            se_ratio=None if block_id == 0 else 8, 
            residual=False if block_id == 0 else True,
        ) for block_id in range(8)])

    def forward(self, x, x_len):

        # Blocks
        x = self.blocks(x)

        # Update Sequence Lengths
        if x_len is not None:
            x_len = (x_len - 1) // 2 + 1
            x_len = (x_len - 1) // 2 + 1

        return x, x_len

###############################################################################
# Modules
###############################################################################

class SqueezeAndExcitationModule(nn.Module):

    """Squeeze And Excitation Module

    Args:
        input_dim: input feature dimension
        reduction_ratio: bottleneck reduction ratio
        inner_act: bottleneck inner activation function

    Input: (batch_size, in_dim, in_length)
    Output: (batch_size, out_dim, out_length)
    
    """

    def __init__(self, input_dim, reduction_ratio, inner_act="relu"):
        super(SqueezeAndExcitationModule, self).__init__()

        assert input_dim % reduction_ratio == 0
        self.conv1 = Conv1d(input_dim, input_dim // reduction_ratio, kernel_size=1)
        self.conv2 = Conv1d(input_dim // reduction_ratio, input_dim, kernel_size=1)

        assert inner_act in ["relu", "swish"]
        if inner_act == "relu":
            self.inner_act = nn.ReLU()
        elif inner_act == "swish":
            self.inner_act = Swish()

    def forward(self, x):

        # Global avg Pooling
        scale = x.mean(dim=-1, keepdim=True)

        # (B, C, 1) -> (B, C // R, 1)
        scale = self.conv1(scale)

        # Inner Act
        scale = self.inner_act(scale)

        # (B, C // R, 1) -> (B, C, 1)
        scale = self.conv2(scale)

        # Sigmoid
        scale = scale.sigmoid()

        # Scale
        x = x * scale

        return x