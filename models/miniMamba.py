#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/10/2024 下午 2:34
# @Author  : 叶航
# @File    : miniMam.py
# @Description : doWhat?
from __future__ import annotations

from typing import Union

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from transformers import AutoTokenizer


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        # B -> b
        # L -> l
        # D -> d_inner or d_in
        # N -> d_state
        # dt_rank 为delta的rank
        super(MambaBlock, self).__init__()
        self.args = args
        
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        
        # x_proj 接收 'x' 并输出特定于输入的 Δ， B， C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj 投影 Δ 从 dt_rank 到 d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
    
    def forward(self, x):
        """
        :param x: shape (b, l, d)
        :return: output: shape (b, l, d)
        """
        # shape (b, l, d_model)
        b, l, d = x.shape
        
        # shape (b, l, d * 2)
        x_res = self.in_proj(x)
        # shape (b, l, d)
        x, res = x_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        
        # shape (b, d, l)
        x = rearrange(x, 'b l d -> b d l')
        # shape (b, d, l)
        x = self.conv1d(x)[:, :, :l]
        # shape (b, l, d)
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        
        return self.out_proj(y)
    
    def ssm(self, x):
        """
        :param x: shape (b, l, d_in)
        :return: output: shape (b, l, d_in)
        """
        d, n = self.A_log.shape
        # Compute ∆ A B C D, the state space parameters.
        # A, D 是独立于输入的 (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        # ∆, B, C 是依赖于输入的 (这是Mamba模型和 linear time invariant S4 的主要区别,这也是为什么Mamba被称为selective state spaces
        
        # float转换为float以便计算
        # A: (d, n)
        A = -torch.exp(self.A_log.float())
        # D: (d, )
        D = self.D.float()
        
        # (b, l, dr + 2 * n)
        x_dbl = self.x_proj(x)
        
        # delta: (b, l, dr)
        # B: (b, l, n)
        # C: (b, l, n)
        delta, B, C = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        
        # TODO: softplus?
        # delta: (b, l, d)
        delta = F.softplus(self.dt_proj(delta))
        # 选择性扫描算法
        # (b, l, d)
        y = self.selective_scan(x, delta, A, B, C, D)
        return y
    
    @staticmethod
    def selective_scan(x, delta, A, B, C, D):
        """
        :param x: shape (b, l, d)
        :param delta: shape (b, l, d)
        :param A: shape (d, n)
        :param B: shape (b, l, n)
        :param C: shape (b, l, n)
        :param D: shape (d, )
        """
        
        b, l, d = x.shape
        n = A.shape[1]
        
        '''
        对连续的参数(A, B)进行离散化
        A 使用零阶保持法(zero-order hold, ZOH)进行离散化 (see Section 2 Equation 4 in the Mamba paper [1])
        B 则使用一种简化的Euler方法进行离散化
        B没有使用ZOH的原因，作者解释如下: "A is the more important term and the performance doesn't change much with the simplification on B"
        '''
        # deltaA: (b, l, d, n)
        # ZOH
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        # 将 delta、B 和 u 张量的对应位置元素相乘，并在最后一个维度上进行求和，输出一个新的张量。
        deltaB_x = einsum(delta, B, x, 'b l d, b l n, b l d -> b l d n')
        
        '''
        执行 selective scan (see scan_SSM() in The Annotated S4 [2])
        # 下面的代码是顺序执行的, 然而在官方代码中使用更快的并行扫描算法实现的(类似于FlashAttention，采用硬件感知扫描)。
        '''
        h = torch.zeros((b, d, n), device=deltaA.device)
        ys = []
        for i in range(l):
            # A!!!!!
            # h(t+1) = A * h(t) + B * x(t)
            # deltaA[:, i]: (b, d, n), h: (b, d, n)
            h = deltaA[:, i] * h + deltaB_x[:, i]
            # y(t) = Cx(t)
            # y: (b, d)
            y = einsum(h, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        # y:(b, l, d)
        y = torch.stack(ys, dim=1)
        y = y + D * x
        
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        # weight: 可学习的参数，调整归一化后的值
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        :param x: shape (b, l, d)
        :return: output: shape (b, l, d)
        RMS的计算步骤：
        Step1: 计算每个样本的均方根值
            Step1.1: 先计算x的平方
            Step1.2: 沿着最后一个维度（通常是特征维度）计算平均值，并加上一个很小的数eps
            Step1.3: 最后取平方根
        Step2: 对每个样本进行归一化
            Step2.1：每个特征值除以其所在样本的均方根值
            Step2.2: 最后乘以可以学习的权重weight,得到最终的输出
        """
        x_squared = x.pow(2)
        x_squared_mean = x_squared.mean(dim=-1, keepdim=True) + self.eps
        x_rms = torch.rsqrt(x_squared_mean)
        
        return x * x_rms * self.weight


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super(ResidualBlock, self).__init__()
        self.args = args
        self.mamba = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
    
    def forward(self, x):
        """

        :param x: shape (b, l, d)
        :return: output: shape (b, l, d)
        """
        # norm -> mamba -> add
        return self.mamba(self.norm(x)) + x


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Mamba, self).__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(self.args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)
        
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        # 权重绑定 weight tying
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids):
        """
        :param input_ids (long tensor): shape (b, l)
        :return: logits: shape (b, l, vocab_size)
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    @staticmethod
    def from_pretrained(pretrained_model_name):
        """
        :param model_path: str
        :return: model
        pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)
        
        return model

























