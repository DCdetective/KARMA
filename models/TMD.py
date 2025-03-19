#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/12/2024 下午 3:57
# @Author  : 叶航
# @File    : TMD.py
# @Description : doWhat?

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from mamba_ssm import Mamba
from layers.KAN_Layer import WaveKANLayer, TaylorKANLayer, JacobiKANLayer
from layers.Embed import DataEmbedding


class AttentionDecomp(nn.Module):
    def __init__(self, enc_in, embed_dim=128):
        super(AttentionDecomp, self).__init__()
        self.dp = nn.Dropout(0.1)
        self.W1 = nn.Linear(enc_in, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)
        self.act = nn.SiLU()
        self.Ws = nn.Linear(embed_dim, enc_in)
        self.Wt = nn.Linear(embed_dim, enc_in)
    
    def forward(self, x):
        # 输入维度调整为 (seq_len, batch, embed_dim)
        x = self.dp(self.W1(x))
        x = x.permute(1, 0, 2)
        trend, _ = self.attention(x, x, x)  # 自注意力生成趋势
        trend = self.act(trend)
        trend = trend.permute(1, 0, 2)  # 恢复 (batch, channels, seq_len)
        residual = x.permute(1, 0, 2) - trend  # 残差部分
        return self.Ws(trend), self.Wt(residual)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class KarmaEncoder(nn.Module):
    def __init__(self, configs, mb_layers, norm_layer=None):
        super(KarmaEncoder, self).__init__()
        self.mb_layers = nn.ModuleList(mb_layers)
        self.time = nn.Linear(configs.d_model, configs.d_model)
        self.norm = norm_layer
    
    def forward(self, x):
        # x [B, L, D]
        for mb_layer in self.mb_layers:
            x = mb_layer(x)
        x_freq = x
        x_time = self.time(x)
        x = x_freq + x_time
        if self.norm is not None:
            x = self.norm(x)
        return x


class KarmaBlock(nn.Module):
    def __init__(self, configs):
        super(KarmaBlock, self).__init__()
        self.d_model = configs.d_model
        
        self.higher = Mamba(
            d_model=configs.d_model // 2 + 2,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        self.lower = TaylorKANLayer(configs.d_model // 2, configs.d_model // 2, order=3, addbias=True)
        
        self.norm = RMSNorm(configs.d_model)
    
    def forward(self, x):
        # x in [B, D, N]
        # x [B, D, F]
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_l = x_freq[..., :x_freq.shape[-1] // 2]
        x_h = x_freq[..., x_freq.shape[-1] // 2:]
        # x [B, D, F * 2] F = N // 2 + 1
        x_l_cat = torch.cat([x_l.real, x_l.imag], dim=-1)
        x_h_cat = torch.cat([x_h.real, x_h.imag], dim=-1)
        # y [B, D, F * 2]
        y_l = self.lower(x_l_cat)
        y_h = self.higher(x_h_cat)
        # y [B, D, F, 2]
        y = torch.cat([y_l, y_h], dim=-1)
        y = y.view(y.shape[0], y.shape[1], -1, 2)
        
        # y = F.softshrink(y, lambd=0.01)
        # y [B, D, F]
        y = torch.view_as_complex(y)
        y = torch.fft.irfft(y, n=self.d_model, dim=-1, norm="ortho")
        
        output = self.norm(y) + x
        return output


class Model1(nn.Module):
    def __init__(self, configs):
        super(Model1, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.karma_decomp = True
        self.norm_layer = RevIN(configs.enc_in)
        if self.karma_decomp:
            self.decompsition = AttentionDecomp(configs.moving_avg)
        else:
            self.decompsition = series_decomp(configs.moving_avg)
        
        self.embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.embedding_s = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.embedding_t = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        
        # self.Ecoder_S = KarmaEncoder(
        #     configs,
        #     mb_layers=[
        #         KarmaBlock(configs) for _ in range(2)
        #     ],
        #     norm_layer=RMSNorm(configs.d_model)
        # )
        self.Ecoder_S = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        
        self.Ecoder_T = Mamba(
            d_model=configs.d_model // 8,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        self.T_prj = nn.Linear(configs.d_model, configs.d_model // 8)
        self.T_back = nn.Linear(configs.d_model // 8, configs.d_model)
        
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    
    def forecast(self, x_enc, x_mark_enc):
        x_enc = self.norm_layer(x_enc, 'norm')
        B, L, D = x_enc.shape
        # x = self.embedding(x_enc, x_mark_enc)
        # [B, L, D]
        seasonal_enc, trend_enc = self.decompsition(x_enc)
        # [B, D, N]
        # embedding
        x_enc = self.embedding(seasonal_enc, None)
        seasonal_enc = self.embedding_s(seasonal_enc, None)
        trend_enc = self.embedding_t(trend_enc, None)
        
        # [B, D, N]
        # Seasonal
        seasonal = self.Ecoder_S(seasonal_enc)
        
        # Trend
        x_t = self.T_prj(trend_enc)
        x_t = self.Ecoder_T(x_t) + x_t
        trend = self.T_back(x_t)
        
        x = seasonal + trend + x_enc
        # x [B, D, L]
        x = self.projection(x).permute(0, 2, 1)
        # x [B, L, D]
        dec_out = self.norm_layer(x, 'denorm')
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        # other tasks not implemented



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        
        self.norm_layer = RevIN(configs.enc_in)
        self.decompsition = series_decomp(configs.moving_avg)
        self.embedding_s = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.embedding_t = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        
        self.Ecoder_S = KarmaEncoder(
            configs,
            mb_layers=[
                KarmaBlock(configs) for _ in range(2)
            ],
            norm_layer=RMSNorm(configs.d_model)
        )
        self.Ecoder_S = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        
        self.Ecoder_T = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    
    def forecast(self, x_enc, x_mark_enc):
        x_enc = self.norm_layer(x_enc, 'norm')
        B, L, D = x_enc.shape
        # x = self.embedding(x_enc, x_mark_enc)
        # seasonal_enc, trend_enc [B, L, D]
        seasonal_enc, trend_enc = self.decompsition(x_enc)
        # seasonal_enc, trend_enc [B, D, N]
        seasonal_enc = self.embedding_s(seasonal_enc, None)
        trend_enc = self.embedding_t(trend_enc, None)
        # seasonal, trend [B, D, N]
        seasonal = self.Ecoder_S(seasonal_enc)
        trend = self.Ecoder_T(trend_enc)
        x = seasonal + trend
        # x [B, D, L]
        x = self.projection(x).permute(0, 2, 1)
        # x [B, L, D]
        dec_out = self.norm_layer(x, 'denorm')
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        # other tasks not implemented