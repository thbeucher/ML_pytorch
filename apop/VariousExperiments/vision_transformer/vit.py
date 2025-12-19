# Taken from https://github.com/lucidrains/vit-pytorch/
# Modified for experiment purpose
import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

##########################################################################################
##                                    HELPERS                                           ##
##########################################################################################
def pair(t):
  return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
  y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
  assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
  omega = torch.arange(dim // 4) / (dim // 4 - 1)
  omega = 1.0 / (temperature ** omega)

  y = y.flatten()[:, None] * omega[None, :]
  x = x.flatten()[:, None] * omega[None, :]
  pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
  return pe.type(dtype)
##########################################################################################

class FeedForward(Module):
  def __init__(self, dim, hidden_dim, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(
      nn.LayerNorm(dim),
      nn.Linear(dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, dim),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Attention(Module):
  def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
    super().__init__()
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.norm = nn.LayerNorm(dim)

    self.attend = nn.Softmax(dim = -1)
    self.dropout = nn.Dropout(dropout)

    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

    self.to_out = nn.Sequential(
      nn.Linear(inner_dim, dim),
      nn.Dropout(dropout)
    ) if project_out else nn.Identity()

  def forward(self, x):
    x = self.norm(x)

    qkv = self.to_qkv(x).chunk(3, dim = -1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    attn = self.attend(dots)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)


class Transformer(Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.layers = ModuleList([])

    for _ in range(depth):
      self.layers.append(ModuleList([
        Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
        FeedForward(dim, mlp_dim, dropout = dropout)
      ]))

  def forward(self, x):
    for attn, ff in self.layers:
      x = attn(x) + x
      x = ff(x) + x
    return self.norm(x)


class ViT(Module):
  def __init__(
    self,
    *,
    image_size,   # int or tuple (H, W): spatial resolution of the input image
    patch_size,   # int or tuple (Ph, Pw): spatial size of each image patch
    dim,          # int: embedding dimension of each patch/token
    depth,        # int: number of Transformer encoder blocks
    heads,        # int: number of attention heads per Transformer block
    mlp_dim,      # int: hidden dimension of the feed-forward (MLP) layer
    channels=3,   # int: number of input image channels (3 for RGB)
    dim_head=64,  # int: dimension of each attention head
    dropout=0.,   # float: dropout rate inside attention and MLP layers
    emb_dropout=0.# float: dropout rate applied to patch embeddings
  ):
    """
    Vision Transformer (ViT) used here as an image-to-image model.

    The model:
    1. Splits the image into non-overlapping patches
    2. Projects each patch into a token embedding
    3. Adds positional embeddings
    4. Processes tokens using a Transformer encoder
    5. Projects tokens back to pixel patches
    6. Reconstructs the image
    """
    super().__init__()
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert image_height % patch_height == 0 and image_width % patch_width == 0,\
      'Image dimensions must be divisible by the patch size.'

    patch_dim = channels * patch_height * patch_width

    self.patchify = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
    self.to_patch_embedding = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))

    self.pos_embedding = posemb_sincos_2d(
      h = image_height // patch_height,
      w = image_width // patch_width,
      dim = dim,
    ) 

    self.dropout = nn.Dropout(emb_dropout)

    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
    
    self.to_patch_pixels = nn.Linear(dim, patch_dim)

    self.unpatchify = Rearrange(
      'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
      h=image_height // patch_height,
      w=image_width // patch_width,
      p1=patch_height,
      p2=patch_width,
      c=channels
    )

  def forward(self, img):
    x = self.patchify(img)  # [B, 3, 32, 32] -> [B, n_patchs, 3*8*8=channels*patch_height*patch_width]
    x = self.to_patch_embedding(x)  # -> [B, n_patchs, dim]

    x = x + self.pos_embedding.to(x.device, dtype=x.dtype)
    x = self.dropout(x)

    x = self.transformer(x)
    
    # project tokens back to pixels
    x = self.to_patch_pixels(x)  # -> [B, n_patchs, 3*8*8=channels*patch_height*patch_width]
    # unpatchify
    x = self.unpatchify(x)  # -> [B, 3, 32, 32]

    return x


if __name__ == '__main__':
  vit = ViT(image_size=16, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=64)
  inp = torch.rand(2, 64, 16, 16)
  out = vit(inp)
  print(f'{out.shape=}')