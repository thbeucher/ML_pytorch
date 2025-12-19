# Taken from https://github.com/lucidrains/vit-pytorch/
# Modified for experiment purpose
import torch

from torch import nn, einsum
from einops import rearrange


def group_dict_by_key(cond, d):
    """
    Splits a dictionary into two dictionaries based on a condition on the key.

    Args:
        cond: function that takes a key and returns True or False
        d: dictionary to split

    Returns:
        (dict_true, dict_false)
    """
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def group_by_key_prefix_and_remove_prefix(prefix, d):
    """
    Extracts all dictionary entries that start with a given prefix,
    removes the prefix from the keys, and returns:
      - extracted dictionary
      - remaining dictionary
    """
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward block implemented with 1x1 convolutions
    for spatial feature maps.
    """
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class DepthWiseConv2d(nn.Module):
    """
    Depthwise separable convolution:
      - Depthwise convolution (per-channel)
      - BatchNorm
      - Pointwise (1x1) convolution
    """
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention using convolutional projections
    instead of linear layers (CvT idea).
    """
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5  # scaling for dot-product attention

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # Query projection (no downsampling)
        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        # Key & Value projection (with downsampling)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        # Reshape for multi-head attention
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale  # Attention scores

        attn = self.attend(dots)  # softmax
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)  # Weighted sum of values
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)  # Restore spatial structure
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Stack of Transformer layers:
      - Attention + residual
      - FeedForward + residual
    """
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim,
                          proj_kernel = proj_kernel,
                          kv_proj_stride = kv_proj_stride,
                          heads = heads,
                          dim_head = dim_head,
                          dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # residual connection
            x = ff(x) + x    # residual connection
        return x


class CvT(nn.Module):
    """
    CvT model with 3 hierarchical stages.
    Each stage:
      - Convolutional embedding
      - LayerNorm
      - Transformer blocks
    """
    def __init__(
        self,
        *,
        s1_emb_dim = 64,            # stage 1 - dimension
        s1_emb_kernel = 5,          # stage 1 - conv kernel - Larger kernel = larger receptive field at early stage
        s1_emb_stride = 2,          # stage 1 - conv stride - Reduces spatial resolution by factor of 2
        s1_proj_kernel = 3,         # stage 1 - attention ds-conv kernel size
                                    # Kernel size of depthwise convolution used to project Q/K/V
                                    # Controls local context captured before attention
        s1_kv_proj_stride = 2,      # stage 1 - attention key / value projection stride
                                    # Stride used when projecting keys and values
                                    # Downsamples K/V to reduce attention cost
        s1_heads = 1,               # stage 1 - heads
        s1_depth = 1,               # stage 1 - depth - Number of Transformer blocks in stage 1
        s1_mlp_mult = 4,            # stage 1 - feedforward expansion factor - Hidden dimension = emb_dim * mlp_mult
        s2_emb_dim = 128,           # stage 2 - (same as above)
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 256,           # stage 3 - (same as above)
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 6,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.,
        channels = 3
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = channels
        layers = []

        for prefix in ('s1', 's2', 's3'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)

            layers.append(nn.Sequential(
                # Patch embedding
                nn.Conv2d(dim,
                          config['emb_dim'],
                          kernel_size = config['emb_kernel'],
                          padding = (config['emb_kernel'] // 2),
                          stride = config['emb_stride']),
                LayerNorm(config['emb_dim']),
                Transformer(dim = config['emb_dim'],
                            proj_kernel = config['proj_kernel'],
                            kv_proj_stride = config['kv_proj_stride'],
                            depth = config['depth'],
                            heads = config['heads'],
                            mlp_mult = config['mlp_mult'],
                            dropout = dropout)
            ))

            dim = config['emb_dim']

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).contiguous() 
            # without .contiguous() it will raise error during loss.backward():
            # RuntimeError: view size is not compatible with input tensor's size and stride
            # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        return x


if __name__ == '__main__':
    vit = CvT(channels=3)
    inp = torch.rand(2, 3, 32, 32)
    out = vit(inp)
    print(f'{out.shape=}')