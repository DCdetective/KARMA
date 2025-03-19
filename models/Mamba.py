import math
from layers.Embed import DataEmbedding_inverted
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from layers.Embed import DataEmbedding


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        
        self.embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # self.embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,configs.dropout)
        self.mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=configs.d_conv,
            expand=configs.expand,
        )
        
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.out_layer = nn.Linear(configs.seq_len, configs.pred_len)
        # self.out_layer = KAN([configs.d_model, configs.c_out])
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
    
    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc
        x_enc = self.embedding(x_enc, None)
        # x = self.embedding(x_enc, x_mark_enc)
        x = self.mamba(x_enc)
        x_out = self.projection(x)
        x_out = self.out_layer(x_out.permute(0, 2, 1)).permute(0, 2, 1)
        x_out = x_out * (std_enc[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        x_out = x_out + (mean_enc[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return x_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['short_term_forecast', 'long_term_forecast', 'stock_prediction']:
            x_out = self.forecast(x_enc, x_mark_enc)
            return x_out[:, -self.pred_len:, :]
        
        # other tasks not implemented