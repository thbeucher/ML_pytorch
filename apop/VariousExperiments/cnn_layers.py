import torch
import random
import torch.nn as nn

from gdn import GDN
from vision_transformer.vit import ViT


class NoiseLayer(nn.Module):
  """
  A layer that conditionally adds Gaussian noise to the input tensor.
  """
  def __init__(self, p=0.5, std=0.1):
    super().__init__()
    self.p = p
    self.std = std

  def forward(self, x, p=None, std=None):
    p = self.p if p is None else p
    std = self.std if std is None else std
    if self.training and p > 0 and std > 0:
      # Check if we should apply noise in this forward pass
      if random.random() < p:
        # Add Gaussian noise
        noise = torch.randn_like(x) * std
        # Add noise and clip the values to the valid [0, 1] range
        noisy_x = torch.clamp(x + noise, 0., 1.)
        return noisy_x
    return x


class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
      nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
      nn.Flatten(),
      nn.Linear(256*4*4, 1),
    )

  def forward(self, x):
    return self.net(x).view(-1)


class CNNEncoder(nn.Module):
  def __init__(self, *args, add_noise=False, noise_p=0.5, noise_std=0.1, add_attn=False,
               norm_first=False, gdn_act=False, **kwargs):
    super().__init__()
    self.down1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1),
                               nn.BatchNorm2d(64) if norm_first else nn.Identity(),
                               GDN(64) if gdn_act else nn.ReLU(True))
    self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1),
                               nn.BatchNorm2d(128),
                               GDN(128) if gdn_act else nn.ReLU(True))
    self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1),
                               nn.BatchNorm2d(256),
                               GDN(256) if gdn_act else nn.ReLU(True))

    self.noise_layer = NoiseLayer(p=noise_p, std=noise_std) if add_noise else nn.Identity()
    
    if add_attn:
      self.attn1 = ViT(image_size=16, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=64)
      self.attn2 = ViT(image_size=8, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=128)
    else:
      self.attn1, self.attn2 = nn.Identity(), nn.Identity()

  def forward(self, x):
    d1 = self.noise_layer(self.down1(x))  # [B, 3, 32, 32] -> [B, 64, 16, 16]
    d1 = self.attn1(d1)
    
    d2 = self.noise_layer(self.down2(d1))  #               -> [B, 128, 8, 8]
    d2 = self.attn2(d2)
    
    d3 = self.noise_layer(self.down3(d2))  #               -> [B, 256, 4, 4]
    
    return d1, d2, d3


class BigCNNBlock(nn.Module):
  def __init__(self, in_c, out_c, se_ratio=4, batch_norm=True):
    super().__init__()
    self.conv3 = nn.Sequential(nn.Conv2d(in_c, out_c, 3, 1, 1),
                               nn.BatchNorm2d(out_c) if batch_norm else nn.Identity(),
                               nn.ReLU(True))
    self.conv5 = nn.Sequential(nn.Conv2d(in_c, out_c, 5, 1, 2),
                               nn.BatchNorm2d(out_c) if batch_norm else nn.Identity(),
                               nn.ReLU(True))
    self.convm = nn.Sequential(nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True))
    self.se = SEBlock(out_c, ratio=se_ratio)
    self.down = nn.Sequential(nn.Conv2d(out_c, out_c, 4, 2, 1), nn.ReLU(True))
  
  def forward(self, x):
    c3 = self.conv3(x)
    c5 = self.conv5(x)
    out = self.convm(c3 + c5)
    out = self.se(out)
    return self.down(out)


class BigCNNEncoder(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.down1 = BigCNNBlock(3, 64, batch_norm=False)
    self.down2 = BigCNNBlock(64, 128)
    self.down3 = BigCNNBlock(128, 256)
  
  def forward(self, x):
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    return d1, d2, d3


class ResDownBlock(nn.Module):
  def __init__(self, in_chan, out_chan, inner_ratio=2, batch_norm=True):
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_chan, int(in_chan*inner_ratio), 3, 1, 1),
                              nn.BatchNorm2d(int(in_chan*inner_ratio)) if batch_norm else nn.Identity(),
                              nn.ReLU(True),
                              nn.Conv2d(int(in_chan*inner_ratio), in_chan, 3, 1, 1),
                              nn.BatchNorm2d(in_chan) if batch_norm else nn.Identity(),
                              nn.ReLU(True))
    self.down = nn.Sequential(nn.Conv2d(in_chan, out_chan, 4, 2, 1), nn.BatchNorm2d(out_chan), nn.ReLU(True))
  
  def forward(self, x):
    out = self.conv(x)  # [B, C, H, W] -> [B, I, H, W] -> [B, C, H, W]
    return self.down(out + x)  # -> [B, O, H/2, W/2]


class ResEncoder(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    self.down1 = ResDownBlock(3, 64, batch_norm=False)
    self.down2 = ResDownBlock(64, 128)
    self.down3 = ResDownBlock(128, 256)
  
  def forward(self, x):
    d1 = self.down1(x)
    d2 = self.down2(d1)
    d3 = self.down3(d2)
    return d1, d2, d3


class SEBlock(nn.Module):
  '''
  The purpose of the Squeeze-and-Excitation (SE) operation is to let a neural network adaptively reweight
  its feature channels based on their global importance for the current input
  -> SE blocks model channel-wise attention

  Standard CNN : Convolutions mix spatial information But all channels are treated equally

  SE blocks introduce a lightweight mechanism to explicitly learn which channels matter more

  “Squeeze”: Global context aggregation
  -> How strongly does this feature channel activate on average across the entire image
  This gives the network global context, which normal convolutions lack

  “Excitation”: Learn channel importance
  -> Forces the network to learn compact inter-channel dependencies
  '''
  def __init__(self, n_channels, ratio=8, inner_chan_min=8, bias=False):
    super().__init__()
    inner_chan = max(n_channels // ratio, inner_chan_min)
    self.squeeze = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]
    self.excit = nn.Sequential(nn.Conv2d(n_channels, inner_chan, 1, bias=bias), nn.ReLU(True),
                               nn.Conv2d(inner_chan, n_channels, 1, bias=bias), nn.Sigmoid())
  
  def forward(self, x):
    out = self.squeeze(x)    # [B, C, H, W] -> [B, C, 1, 1]
    excit = self.excit(out)  # [B, C, 1, 1] -> [B, C//ratio, 1, 1] -> [B, C, 1, 1]
    return x * excit  # channel-wise reweighting


class SEGEncoder(nn.Module):
  def __init__(self, *args, groups=1, norm_first=False, **kwargs):
    super().__init__()
    self.down1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1),
                               nn.BatchNorm2d(64) if norm_first else nn.Identity(),
                               nn.ReLU(True))
    self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, groups=groups), nn.BatchNorm2d(128), nn.ReLU(True))
    self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, groups=groups), nn.BatchNorm2d(256), nn.ReLU(True))
    self.se1 = SEBlock(64)
    self.se2 = SEBlock(128)
    self.se3 = SEBlock(256)

  def forward(self, x):
    d1 = self.se1(self.down1(x))
    d2 = self.se2(self.down2(d1))
    d3 = self.se3(self.down3(d2))
    return d1, d2, d3


class CNNDecoder(nn.Module):
  def __init__(self, add_attn=True, gdn_act=False):
    super().__init__()
    self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1),
                             nn.BatchNorm2d(128),
                             GDN(128) if gdn_act else nn.ReLU(True))
    self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1),
                             nn.BatchNorm2d(64),
                             GDN(64) if gdn_act else nn.ReLU(True))
    self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    if add_attn:
      self.attn1 = ViT(image_size=8, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=128)
      self.attn2 = ViT(image_size=16, patch_size=4, dim=64, depth=2, heads=4, mlp_dim=128, dim_head=32, channels=64)
    else:
      self.attn1, self.attn2 = nn.Identity(), nn.Identity()
  
  def forward(self, d3, d2=None, d1=None, return_intermediate=False):
    u1 = self.up1(d3)   #                -> [B, 128, 8, 8]
    u1 = self.attn1(u1)

    if d2 is not None:  # skip connection
      u1 = u1 + d2

    u2 = self.up2(u1)   #                -> [B, 64, 16, 16]
    u2 = self.attn2(u2)

    if d1 is not None:  # skip connection
      u2 = u2 + d1

    u3 = self.up3(u2)  #                -> [B, 3, 32, 32]

    if return_intermediate:
      return u1, u2, u3
    return u3


class FiLM(nn.Module):
  # FiLM: Visual Reasoning with a General Conditioning Layer
  # https://arxiv.org/pdf/1709.07871
  def __init__(self, cond_dim, channels):
    super().__init__()
    self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * channels))
    # important for stability
    nn.init.zeros_(self.mlp[-1].weight)
    nn.init.zeros_(self.mlp[-1].bias)

  def forward(self, h, cond):
    gamma, beta = self.mlp(cond).chunk(2, dim=1)
    gamma = gamma[:, :, None, None]
    beta  = beta[:, :, None, None]
    return h * (1 + gamma) + beta


class ResidualFullBlock(nn.Module):
  def __init__(self, in_chan, out_chan, groups=1, norm_first=True, cond_emb=None,
               use_attn=True, attn_ratio=8,
               pooling=False, pool_opt='cnn',      # 'cnn' or 'avg'
               upscaling=False, upscale_opt='cnn'):  # 'cnn' or 'upsample'
    super().__init__()
    self.conv = nn.Sequential(nn.Conv2d(in_chan, out_chan, 3, 1, 1, groups=groups),
                               nn.BatchNorm2d(out_chan) if norm_first else nn.Identity(),
                               nn.SiLU(),
                               nn.Conv2d(out_chan, out_chan, 3, 1, 1, groups=groups),
                               nn.BatchNorm2d(out_chan) if norm_first else nn.Identity())
    
    self.film = FiLM(cond_emb, out_chan) if cond_emb is not None else nn.Identity()
    
    self.skip_connection = nn.Conv2d(in_chan, out_chan, 1, groups=groups) if in_chan != out_chan else nn.Identity()
    
    self.chan_attn = SEBlock(out_chan, ratio=attn_ratio) if use_attn else nn.Identity()
    
    self.act = nn.SiLU()
    
    if pooling:
      self.pool = nn.Sequential(nn.Conv2d(out_chan, out_chan, 4, 2, 1, groups=groups),
                                nn.BatchNorm2d(out_chan), nn.SiLU())\
        if pool_opt == 'cnn' else nn.AvgPool2d(2)
    else:
      self.pool = nn.Identity()
    
    if upscaling:
      self.upscale = nn.Sequential(nn.ConvTranspose2d(out_chan, out_chan, 4, 2, 1, groups=groups),
                                   nn.BatchNorm2d(out_chan), nn.SiLU())\
        if upscale_opt == 'cnn' else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    else:
      self.upscale = nn.Identity()
  
  def forward(self, x, c=None):                        # x: [B, Ci, H, W] | t: [B, time_dim]
    out = self.conv(x)                                 # -> [B, Co, H, W]
    out = self.chan_attn(out)

    if c is not None:
      out = self.film(out, c)
    out = self.act(out + self.skip_connection(x))

    out = self.pool(out)                               # -> [B, Co, H/2, W/2] if pooling
    out = self.upscale(out)                            # -> [B, Co, H*2, W*2] if upscaling
    return out


class EnhancedResidualFullBlock(nn.Module):
  def __init__(self, in_chan, out_chan, groups=1, norm_first=True, cond_emb=None,
               use_attn=True, attn_ratio=8,
               pooling=False, pool_opt='cnn',
               upscaling=False, upscale_opt='cnn'):
    super().__init__()
    self.rfb = ResidualFullBlock(in_chan=in_chan, out_chan=out_chan, groups=groups,
                                 norm_first=norm_first, cond_emb=cond_emb,
                                 use_attn=use_attn, attn_ratio=attn_ratio,
                                 pooling=pooling, pool_opt=pool_opt,
                                 upscaling=upscaling, upscale_opt=upscale_opt)
    self.refine = nn.Sequential(nn.Conv2d(out_chan, out_chan, 1, groups=groups), nn.BatchNorm2d(out_chan), nn.SiLU())

  def forward(self, x, c=None):
    out = self.rfb(x, c=c)
    return self.refine(out)


CNN_LAYERS = {'NoiseLayer': NoiseLayer, 'Critic': Critic, 'CNNEncoder': CNNEncoder, 'CNNDecoder': CNNDecoder,
              'BigCNNBlock':BigCNNBlock, 'BigCNNEncoder': BigCNNEncoder, 'ResDownBlock': ResDownBlock,
              'ResEncoder': ResEncoder, 'SEBlock': SEBlock, 'SEGEncoder': SEGEncoder,
              'EnhancedResidualFullBlock': EnhancedResidualFullBlock}


if __name__ == '__main__':
  print(f'Number of trainable parameters:')
  for k, v in CNN_LAYERS.items():
    if 'Encoder' in k:
      n_trainable_params = sum(p.numel() for p in v().parameters() if p.requires_grad)
      print(f'{k:<15}: {n_trainable_params:,}')