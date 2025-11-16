phi-mir0-opt 

import argparse
import math
import json
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

# --------------------------- PhiAdam ---------------------------
class PhiAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_inv = 1.0 / phi
        defaults = dict(lr=lr, betas=(betas[0] * phi_inv, betas[1] * phi_inv), eps=eps, weight_decay=weight_decay)
        super(PhiAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PhiAdam does not support sparse gradients')
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / torch.sqrt(torch.tensor(bias_correction2, device=exp_avg.device))).add_(group['eps'])
                step_size = group['lr'] / torch.sqrt(torch.tensor(bias_correction1, device=exp_avg.device))
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

# --------------------------- PhiMultiHeadSelfAttention ---------------------------
class PhiMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        head_weights = np.array([phi ** (-i) for i in range(n_heads)], dtype=np.float32)
        head_weights = head_weights / head_weights.sum()
        self.register_buffer('base_head_weights', torch.tensor(head_weights).view(1, n_heads, 1, 1))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask=None):
        B, T, _ = x.size()
        q = self.q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        head_w = self.base_head_weights * self.scale
        out = out * head_w
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(out)

# --------------------------- Transformer components ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = PhiMultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.attn(x, mask=mask)
        x = x + self.dropout(x2)
        x = self.norm1(x)
        x2 = self.ff(x)
        x = x + self.dropout(x2)
        x = self.norm2(x)
        return x

class CharTransformerLM(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=256, n_heads=8, n_layers=6, dim_feedforward=10
