import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, qkv_bias=False, qk_scale=None, dropout_rate=0.0) -> None:
