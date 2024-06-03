#
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Designed for input of shape [batch_size, seq_len, d_model]
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeatTimeTransformer(nn.Module):
    def __init__(self, args):
        super(FeatTimeTransformer, self).__init__()
        self.args = args
        self.classes = args.vocab_size
        self.proj = nn.Linear(args.input_dim, args.transformer_dim)
        self.ln = nn.LayerNorm(args.transformer_dim)
        self.pos_encoder = PositionalEncoding(args.transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=args.transformer_dim,
                                                     nhead=args.transformer_heads,
                                                     dropout=args.transformer_dropout,
                                                     batch_first=True),
            num_layers=args.transformer_layers)
        self.head = nn.Linear(args.transformer_dim, self.classes)

    def forward(self, input, middle=False):
        x = self.ln(self.proj(input))
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        if middle:
            return x
        x = self.head(x)
        x = x.view(-1, self.classes)
        return x