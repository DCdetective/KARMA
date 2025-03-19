import torch
from mamba_ssm import Mamba

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim,  # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,  # Local convolution width
    expand=2,  # Block expansion factor
).to("cuda")
y = model(x)
print("Mamba result", y.shape)
assert y.shape == x.shape

import torch
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 512
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    # make sure d_model * expand / headdim = multiple of 8
    d_model=dim,  # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,  # Local convolution width
    expand=2,  # Block expansion factor
    headdim=64,  # default 64
).to("cuda")
y = model(x)
print("Mamba2 result", y.shape)
assert y.shape == x.shape
