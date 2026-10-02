# -*- coding: utf-8 -*-
"""
model.py — Lightweight multi-scale CNN for circRNA-RBP interaction prediction.

Architecture (80,833 trainable parameters):
  Branch A (sequence)  : one-hot (B, L, 4) -> multi-scale Conv1d (k=4/8/16, 64 ch each)
                         -> global max pooling -> concat -> BN -> (B, 192)
  Branch B (K-mer)     : 3-mer(64) + 4-mer(256) frequency vector (B, 320)
                         -> BN -> FC(64) -> BN -> (B, 64)
  Branch C (structure) : Nussinov 5-channel encoding (B, L, 5)
                         -> Conv1d (k=4/8, 32 ch each) -> global max pooling
                         -> concat -> BN -> (B, 64)
  Fusion head          : concat(192+64+64=320) -> BN -> FC(128) -> FC(64) -> FC(1)

The module is shared by train_full.py, evaluate.py and eval_per_rbp.py so that
training and inference always use an identical graph.
"""

import torch
import torch.nn as nn

MAX_LEN = 501      # padded/truncated circRNA sequence length
DROPOUT = 0.3


class CNNGPUModel(nn.Module):
    """Three-branch multi-scale CNN with K-mer and secondary-structure branches."""

    def __init__(self, kmer_dim=320, struct_dim=5, dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()
        self.max_len = max_len

        # Branch A: multi-scale sequence CNN
        self.conv4 = nn.Conv1d(4, 64, kernel_size=4, padding=2)
        self.conv8 = nn.Conv1d(4, 64, kernel_size=8, padding=4)
        self.conv16 = nn.Conv1d(4, 64, kernel_size=16, padding=8)
        self.bn_seq = nn.BatchNorm1d(192)

        # Branch B: K-mer frequency
        self.kmer_bn = nn.BatchNorm1d(kmer_dim)
        self.kmer_fc = nn.Linear(kmer_dim, 64)
        self.kmer_bn2 = nn.BatchNorm1d(64)

        # Branch C: secondary structure
        self.conv_s4 = nn.Conv1d(struct_dim, 32, kernel_size=4, padding=2)
        self.conv_s8 = nn.Conv1d(struct_dim, 32, kernel_size=8, padding=4)
        self.bn_strc = nn.BatchNorm1d(64)

        # Fusion head
        self.fc1 = nn.Linear(192 + 64 + 64, 128)
        self.bn_f1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, seq, kmer, struct):
        """seq (B, L, 4), kmer (B, 320), struct (B, L, 5) -> logits (B,)"""
        L = self.max_len

        # Branch A
        x = seq.permute(0, 2, 1)                       # (B, 4, L)
        c4 = self.relu(self.conv4(x))[:, :, :L]
        c8 = self.relu(self.conv8(x))[:, :, :L]
        c16 = self.relu(self.conv16(x))[:, :, :L]
        a = torch.cat([c4.max(2).values, c8.max(2).values, c16.max(2).values], dim=1)
        a = self.drop(self.bn_seq(a))

        # Branch B
        b = self.relu(self.kmer_fc(self.kmer_bn(kmer)))
        b = self.drop(self.kmer_bn2(b))

        # Branch C
        s = struct.permute(0, 2, 1)                    # (B, 5, L)
        cs4 = self.relu(self.conv_s4(s))[:, :, :L]
        cs8 = self.relu(self.conv_s8(s))[:, :, :L]
        c = torch.cat([cs4.max(2).values, cs8.max(2).values], dim=1)
        c = self.drop(self.bn_strc(c))

        # Fusion
        x = torch.cat([a, b, c], dim=1)
        x = self.drop(self.relu(self.bn_f1(self.fc1(x))))
        x = self.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
