import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/VainF/pytorch-msssim
sys.path.append('../../../pytorch-msssim/')
sys.path.append('../../../pytorch-msssim/tests/ae_example/models/')
from gdn import GDN  # GDN(n_features) | GDN(n_features, inverse=True)
from pytorch_msssim.ssim import MS_SSIM, ms_ssim


# criterion = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
class MS_SSIM_Loss(MS_SSIM):
  def forward(self, img1, img2):
    return 100*(1 - super(MS_SSIM_Loss, self).forward(img1, img2))


class ImageEmbedder(nn.Module):
  def __init__(self, n_input_features=3, output_emb_size=256, gdn_act=False):
    super().__init__()
    # # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    self.embedder = nn.Sequential(
      nn.Conv2d(n_input_features, 16, 4, stride=2, padding=1),  # [3, 180, 180] -> [16, 90, 90]
      GDN(16) if gdn_act else nn.ReLU(),
      nn.Conv2d(16, 32, 4, stride=2, padding=1),  # [16, 90, 90] -> [32, 45, 45]
      GDN(32) if gdn_act else nn.ReLU(),
      nn.Conv2d(32, 64, 4, stride=2, padding=1),  # [32, 45, 45] -> [64, 22, 22]
      GDN(64) if gdn_act else nn.ReLU(),
      nn.Conv2d(64, 128, 5, stride=2, padding=1),  # [64, 22, 22] -> [128, 10, 10]
      GDN(128) if gdn_act else nn.ReLU(),
      nn.Conv2d(128, output_emb_size, 10),  # [128, 10, 10] -> [256, 1, 1]
    )
  
  def forward(self, img):  # [bs, 3, 180, 180]
    return self.embedder(img).squeeze()


class ImageReconstructor(nn.Module):
  def __init__(self, n_input_features=256, n_output_features=3, gdn_act=False):
    super().__init__()
    # in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0
    self.reconstructor = nn.Sequential(
      nn.ConvTranspose2d(n_input_features, 128, 10),  # [256, 1, 1] -> [128, 10, 10]
      GDN(128, inverse=True) if gdn_act else nn.ReLU(),
      nn.ConvTranspose2d(128, 64, 5, stride=2, padding=1, output_padding=1),  # [128, 10, 10] -> [64, 22, 22]
      GDN(64, inverse=True) if gdn_act else nn.ReLU(),
      nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1),  # [64, 22, 22] -> [32, 45, 45]
      GDN(32, inverse=True) if gdn_act else nn.ReLU(),
      nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0),  # [32, 45, 45] -> [16, 90, 90]
      GDN(16, inverse=True) if gdn_act else nn.ReLU(),
      nn.ConvTranspose2d(16, n_output_features, 4, stride=2, padding=1, output_padding=0)  # -> [3, 180, 180]
    )
  
  def forward(self, x):  # [bs, 256]
    batch_size, emb_dim = x.shape
    return torch.sigmoid(self.reconstructor(x.view(batch_size, emb_dim, 1, 1)))


if __name__ == '__main__':
  ie = ImageEmbedder()
  encoded_img = ie(torch.randn(2, 3, 180, 180))
  print(f'encoded_img={encoded_img.shape}')

  ir = ImageReconstructor()
  print(ir(encoded_img).shape)