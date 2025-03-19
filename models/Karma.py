import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_inverted
from layers.RevIN import RevIN
from layers.FAN import FAN
from layers.SAN import SAN
from mamba_ssm import Mamba
from layers.KAN_Layer import WaveKANLayer, TaylorKANLayer, JacobiKANLayer
from layers.Embed import DataEmbedding
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class Karma_decomp(nn.Module):
    def __init__(self, enc_in, embed_dim=128):
        super(Karma_decomp, self).__init__()
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
        seasonal = x.permute(1, 0, 2) - trend  # 残差部分
        return self.Ws(seasonal), self.Wt(trend)


class FFN(nn.Module):
    def __init__(self, configs, d_model, d_ff):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = configs.dropout
        
        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.d_ff, out_channels=self.d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, x):
        y = x = self.norm1(x)
        y = self.dropout(F.silu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class KarmaEncoder(nn.Module):
    def __init__(self, configs, km_layers, norm_layer=None):
        super(KarmaEncoder, self).__init__()
        decompose_layer = 1
        wave = 'db4'
        
        mode = 'symmetric'
        self.WT = DWT1DForward(wave=wave, J=decompose_layer, mode=mode)
        self.IWT = DWT1DInverse(wave=wave)
        self.km_layers = nn.ModuleList(km_layers)
        
        self.norm = norm_layer
    
    def forward(self, x):
        yl, yhs = self.WT(x)
        xl, xhs = yl, yhs[0]
        for km_layer in self.km_layers:
            x, xl, xhs = km_layer(x, xl, xhs)
        x_out = self.IWT((xl, [xhs])) + x
        if self.norm is not None:
            x_out = self.norm(x_out)
        return x_out


class KarmaBlock(nn.Module):
    def __init__(self, configs):
        super(KarmaBlock, self).__init__()
        self.d_model = configs.d_model
        self.d_state = configs.d_state
        
        self.freq1 = Mamba(
            d_model=(configs.d_model + 7) // 2,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        # self.T_prj = nn.Linear(configs.d_model, configs.d_model // 4)
        # self.T_back = nn.Linear(configs.d_model // 4, configs.d_model)
        self.freq2_g = TaylorKANLayer(configs.d_model // 2, configs.d_model // 2, order=3, addbias=True)
        self.freq2 = Mamba(
            d_model=(configs.d_model + 7) // 2,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        
        self.time1 = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        self.time2 = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        ) if not configs.use_decomp else self.time1
        self.norm = RMSNorm(configs.d_model)
        self.norm1 = RMSNorm(configs.d_model)
    
    def forward(self, x, x1, x2):
        x_freq1 = self.freq1(x1)
        x_freq2 = self.freq2(x2)
        # x_freqs = torch.cat([x_freq1, x_freq2], dim=-1)
        x_time = self.time1(self.norm(x)) + self.time2(self.norm1(x).flip(dims=[1])).flip(dims=[1]) + x
        
        return x_time, x_freq1, x_freq2


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.karma_decomp = False
        self.norm_layer = RevIN(configs.enc_in)
        if self.configs.use_norm:
            if self.configs.norm_method == 'RevIN':
                self.norm_layer = RevIN(self.configs.enc_in)
            elif self.configs.norm_method == 'FAN':
                self.norm_layer = FAN(self.configs.seq_len, self.configs.pred_len, self.configs.enc_in)
            elif self.configs.norm_method == 'SAN':
                self.norm_layer = SAN(self.configs.seq_len, self.configs.pred_len, self.configs.enc_in)
        
        if self.karma_decomp:
            self.decompsition = Karma_decomp(configs.enc_in, self.configs.embed_dim)
        else:
            self.decompsition = series_decomp(configs.moving_avg)
        
        self.embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                configs.dropout)
        self.embedding_s = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.embedding_t = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        
        self.Ecoder_S = KarmaEncoder(
            configs,
            km_layers=[
                KarmaBlock(configs) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        # self.Ecoder_S = Mamba(
        #     d_model=configs.d_model,
        #     d_state=configs.d_state,
        #     d_conv=configs.d_conv,
        #     expand=configs.expand,
        # )
        
        self.Ecoder_T = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        # self.T_prj = nn.Linear(configs.d_model, configs.d_model // 8)
        # self.T_back = nn.Linear(configs.d_model // 8, configs.d_model)
        # self.Ecoder_T = KarmaEncoder(
        #     configs,
        #     km_layers=[
        #         KarmaBlock(configs) for _ in range(configs.e_layers)
        #     ],
        #     norm_layer=nn.LayerNorm(configs.d_model)
        # )
        # self.Ecoder_T = WaveKANLayer(configs.d_model, configs.d_model, wavelet_type="mexican_hat", device="cuda")
        
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.get_parameter_number()
    
    def get_parameter_number(self):
        """
        Number of model parameters (without stable diffusion)
        """
        total_num = sum(p.numel() for p in self.parameters())
        param_memory = total_num * 4 / 1024 / 1024
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_ratio = trainable_num / total_num
        
        print('total_num:', total_num)
        print('param_memory:', param_memory)
        print('trainable_num:', total_num)
        print('trainable_ratio:', trainable_ratio)
    
    def forecast(self, x, x_mark):
        if self.configs.use_norm:
            if self.configs.norm_method == 'SAN':
                x, s = self.norm_layer(x, 'norm')
            elif self.configs.norm_method == 'NS':
                means = x.mean(1, keepdim=True).detach()
                x = x - means
                stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x /= stdev
            else:
                x = self.norm_layer(x, 'norm')
        
        B, L, D = x.shape
        if self.configs.use_decomp:
            seasonal_enc, trend_enc = self.decompsition(x)
            seasonal_enc = self.embedding_s(seasonal_enc, None)
            trend_enc = self.embedding_t(trend_enc, None)
            # print(trend_enc.shape)
            x_enc_t = self.Ecoder_T(trend_enc)
            # x_t = self.T_prj(trend_enc)
            # x_t = self.Ecoder_T(x_t)
            # x_enc_t = self.T_back(x_t)
        
        # embedding
        x_enc_s = self.Ecoder_S(self.embedding(x, x_mark) if not self.configs.use_decomp else seasonal_enc)
        
        # x = torch.concat([seasonal + trend, x_enc], dim=-1)
        if self.configs.use_decomp:
            x = x_enc_s + x_enc_t
        else:
            x = x_enc_s
        # x [B, D, L]
        x = self.projection(x).permute(0, 2, 1)[:, :, :D]
        # x [B, L, D]
        if self.configs.use_norm:
            if self.configs.norm_method == 'SAN':
                x = self.norm_layer(x, 'denorm', s)
            elif self.configs.norm_method == 'NS':
                x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
                x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            else:
                x = self.norm_layer(x, 'denorm')
        
        return x
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        # other tasks not implemented

