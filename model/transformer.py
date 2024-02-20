import torch.nn as nn
import torch


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        B,N,C = x.shape
        qkv_ip = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        )
        # print(qkv_ip.shape)
        qkv = qkv_ip.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = attention.softmax(dim=-1)
        attn = self.attn_drop(attention)

        x = (attention @ value).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))

    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class IntermediateSequential(nn.Module):
    def __init__(self, *args, return_intermediate=True):
        super().__init__()
        self.args = args
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList(args)

    def forward(self, x):
        if not self.return_intermediate:
            output = x
            for layer in self.layers:
                output = layer(output)
            return output
        intermediate_outputs = {}
        output = x

        for i, layer in enumerate(self.layers):
            output = intermediate_outputs[str(i)] = layer(output)

        return output, intermediate_outputs


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        return_intermediate=True,
    ):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )

        self.net = IntermediateSequential(*layers, return_intermediate=return_intermediate)


    def forward(self, x):
        return self.net(x)


    
if __name__ == "__main__":


    dim = 512
    depth = 6
    heads = 8
    mlp_dim = 2048
    dropout_rate = 0.1
    attn_dropout_rate = 0.1
    model = TransformerModel(dim, depth, heads, mlp_dim, dropout_rate, attn_dropout_rate)

    batch_size = 4
    sequence_length = 10
    input_tensor = torch.rand(batch_size, sequence_length, dim)

    output, intermediate_outputs = model(input_tensor)

    print("Output shape:", output.shape)