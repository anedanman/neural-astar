"""Encoder module
Author: Ryo Yonetani
Affiliation: OSX
"""

from typing import Optional

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.attention import SpatialTransformer
from modules.pos_emb import PosEmbeds


class EncoderBase(nn.Module):
    def __init__(self,
                 input_dim: int,
                 encoder_depth: int = 4,
                 const: Optional[float] = None):
        """
        Base Encoder

        Args:
            input_dim (int): input dimension
            encoder_depth (int, optional): depth of the encoder. Defaults to 4.
            const (Optional[float], optional): learnable weight to be multiplied for h(v). Defaults to None.
        """
        super().__init__()
        self.model = self.construct_encoder(input_dim, encoder_depth)
        if const is not None:
            self.const = nn.Parameter(torch.ones(1) * const)
        else:
            self.const = 1.

    def construct_encoder(self, input_dim, encoder_depth) -> nn.Module:
        pass

    def forward(self, x):
        # y = torch.sigmoid(self.model(x))
        y = (self.model(x) + 1) / 2
        return y * self.const


class Unet(EncoderBase):

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def construct_encoder(self, input_dim: int,
                          encoder_depth: int) -> nn.Module:
        """
        Unet encoder

        Args:
            input_dim (int): input dimension
            encoder_depth (int, optional): depth of the encoder.
        """
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        return smp.Unet(
            encoder_name="vgg16_bn",
            encoder_weights=None,
            classes=1,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )


class CNN(EncoderBase):

    CHANNELS = [32, 64, 128, 256]

    def construct_encoder(self, input_dim: int,
                          encoder_depth: int) -> nn.Module:
        """
        Simple CNN encoder

        Args:
            input_dim (int): input dimension
            encoder_depth (int, optional): depth of the encoder.
        """
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [1]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
        # return nn.Sequential(*blocks[:-1])
        return Autoencoder(in_channels=4, resolution=(128, 128))
    

class Autoencoder(nn.Module):
    def __init__(self, 
                in_channels=2, 
                out_channels=1, 
                hidden_channels=64,
                attn_blocks=4,
                attn_heads=4,
                cnn_dropout=0.15,
                attn_dropout=0.15,
                downsample_steps=3, 
                resolution=(64, 64),
                *args,
                **kwargs):
        super().__init__()
        heads_dim = hidden_channels // attn_heads
        self.encoder = Encoder(in_channels, hidden_channels, downsample_steps, cnn_dropout)
        self.pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.transformer = SpatialTransformer(
            hidden_channels, 
            attn_heads,
            heads_dim,
            attn_blocks, 
            attn_dropout
        )
        self.decoder_pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.decoder = Decoder(hidden_channels, out_channels, downsample_steps, cnn_dropout)
                
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        return x
