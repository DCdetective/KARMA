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
        return self.Ws(residual), self.Wt(trend)

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
    def __init__(self, configs, mb_layers, norm_layer=None):
        super(KarmaEncoder, self).__init__()
        decompose_layer = 1
        wave = 'haar'
        
        mode = 'symmetric'
        self.WT = DWT1DForward(wave=wave, J=decompose_layer, mode=mode)
        self.IWT = DWT1DInverse(wave=wave)
        self.mb_layers = nn.ModuleList(mb_layers)
        
        self.norm = norm_layer
    
    def forward(self, x):
        # yl, yhs = self.WT(x)
        # xl, xhs = yl, yhs[0]
        for mb_layer in self.mb_layers:
            x, xl, xhs = mb_layer(x, x, x)
        x_out = x
        if self.norm is not None:
            x_out = self.norm(x_out)
        return x_out





class KarmaBlock(nn.Module):
    def __init__(self, configs):
        super(KarmaBlock, self).__init__()
        self.d_model = configs.d_model
        self.d_state = configs.d_state
        
        self.freq1 = Mamba(
            d_model=configs.d_model // 2,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        # self.T_prj = nn.Linear(configs.d_model, configs.d_model // 4)
        # self.T_back = nn.Linear(configs.d_model // 4, configs.d_model)
        # self.time = TaylorKANLayer(configs.d_model, configs.d_model, order=3, addbias=True)
        self.freq2 = Mamba(
            d_model=configs.d_model // 2,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        
        self.time = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        
        self.norm1 = RMSNorm(configs.d_model)
        self.norm2 = RMSNorm(configs.d_model)
    
    def forward(self, x, x1, x2):
        # x_freq1 = self.freq1(x1)
        # x_freq2 = self.freq2(x2)
        # x_freqs = torch.cat([x_freq1, x_freq2], dim=-1)
        output = self.time(self.norm1(x)) + x
        # x_freq1 = self.ffn_freq1(x_freq1)
        # x_freq2 = self.ffn_freq2(x_freq2)
        return output, x, x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.karma_decomp = True
        self.norm_layer = RevIN(configs.enc_in)
        if self.configs.use_norm:
            if self.configs.norm_method == 'RevIN':
                self.norm_layer = RevIN(self.configs.enc_in)
            elif self.configs.norm_method == 'FAN':
                self.norm_layer = FAN(self.configs.seq_len, self.configs.pred_len, self.configs.enc_in)
            elif self.configs.norm_method == 'SAN':
                self.norm_layer = SAN(self.configs.seq_len, self.configs.pred_len, self.configs.enc_in)
        if self.karma_decomp:
            self.decompsition = AttentionDecomp(configs.enc_in)
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
            mb_layers=[
                KarmaBlock(configs) for _ in range(4)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        # self.Ecoder_S = Mamba(
        #     d_model=configs.d_model,
        #     d_state=configs.d_state,
        #     d_conv=configs.d_conv,
        #     expand=configs.expand,
        # )
        
        # self.Ecoder_T = Mamba(
        #     d_model=configs.d_model // 4,
        #     d_state=configs.d_state,
        #     d_conv=configs.d_conv,
        #     expand=configs.expand,
        # )
        # self.T_prj = nn.Linear(configs.d_model, configs.d_model // 4)
        # self.T_back = nn.Linear(configs.d_model // 4, configs.d_model)
        self.Ecoder_T = KarmaEncoder(
            configs,
            mb_layers=[
                KarmaBlock(configs) for _ in range(configs.e_layers + 2)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    
    def forecast(self, x, x_mark):
        if self.configs.use_norm:
            if self.configs.norm_method == 'SAN':
                x, s = self.norm_layer(x, 'norm')
            else:
                x = self.norm_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
        B, L, D = x.shape
        # x = self.embedding(x_enc, x_mark_enc)
        # [B, L, D]
        # seasonal_enc, trend_enc = self.decompsition(x)
        # [B, D, N]
        # embedding
        x = self.embedding(x, x_mark)
        # seasonal_enc = self.embedding_s(seasonal_enc, x_mark)
        # trend_enc = self.embedding_t(trend_enc, x_mark)
        
        # [B, D, N]
        # # Seasonal
        # seasonal = self.Ecoder_S(seasonal_enc)
        #
        # # Trend
        # x_t = self.T_prj(trend_enc)
        # trend = self.Ecoder_T(trend_enc)
        # trend = self.T_back(x_t)
        
        # x = torch.concat([seasonal + trend, x_enc], dim=-1)
        # x = seasonal + trend + x
        # x [B, D, L]
        x = self.Ecoder_S(x)
        x = self.projection(x).permute(0, 2, 1)[:, :, :D]
        # x [B, L, D]
        if self.configs.use_norm:
            if self.configs.norm_method == 'SAN':
                x = self.norm_layer(x, 'denorm', s)
            else:
                x = self.norm_layer(x, 'denorm')
        else:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        return x
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        # other tasks not implemented


class Model1(nn.Module):
    def __init__(self, configs):
        super(Model1, self).__init__()
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
                KarmaBlock(configs) for _ in range(configs.e_layers)
            ],
            norm_layer=RMSNorm(configs.d_model)
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