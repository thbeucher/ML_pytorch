import torch
import torch.nn as nn


def compute_out_conv(size_in, kernel=3, stride=1, padding=0, dilation=1):
  return (size_in + 2 * padding - dilation * (kernel - 1) -1) // stride + 1


def load_model_from_NaiveConvED(model, filename):
  checkpoint = torch.load(filename)
  count_loaded = 0
  model_state_dict = {}
  for k1, v1 in model.state_dict().items():
    k = [k2 for k2 in checkpoint['model_state_dict'] if k1 in k2]
    if len(k) > 0:
      model_state_dict[k1] = checkpoint['model_state_dict'][k[0]]
      count_loaded += 1
    else:
      model_state_dict[k1] = v1
  model.load_state_dict(model_state_dict)
  print(f'{count_loaded}/{len(model.state_dict())} tensors loaded.')
  

class NaiveConvEncoder(nn.Module):
  def __init__(self, n_feats=100, in_size=400, out_size=80):
    super().__init__()
    self.relu = nn.ReLU(inplace=True)

    self.c1 = nn.Conv2d(1, n_feats, 7, padding=3)
    self.c2 = nn.Conv2d(n_feats, n_feats, 5, padding=2)
    self.c3 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
    # identity/residual
    self.c4 = nn.Conv2d(n_feats, n_feats, 3)
    self.c5 = nn.Conv2d(n_feats, n_feats, 3)
    self.c6 = nn.Conv2d(n_feats, n_feats, 3)
    
    self.mp = nn.MaxPool2d(2, return_indices=True)

    self.c7 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
    self.c8 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
    # identity/residual
    self.c9 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
    self.c10 = nn.Conv2d(n_feats, n_feats, 3, padding=1)

    self.in_size = compute_out_conv(compute_out_conv(compute_out_conv(in_size))) // 2
    self.final_proj = nn.Linear(self.in_size * n_feats, out_size)
  
  def forward(self, x):
    out = self.relu(self.c1(x))
    out = self.relu(self.c2(out))
    out = self.relu(self.c3(out))

    out = out + x  # identity/residual

    out = self.relu(self.c4(out))
    out = self.relu(self.c5(out))
    out = self.relu(self.c5(out))

    out, idx = self.mp(out)
    x = out

    out = self.relu(self.c7(out))
    out = self.relu(self.c8(out))

    out = out + x  # identity/residual

    out = self.relu(self.c9(out))
    out = self.relu(self.c10(out))

    bs, _, sl, _ = out.shape
    out = self.relu(self.final_proj(out.reshape(bs, 1, sl, -1)))
    return out, idx


class NaiveConvDecoder(nn.Module):
  def __init__(self, n_feats=100, in_size=80, out_size=197):
    super().__init__()
    self.n_feats = n_feats

    self.relu = nn.ReLU(inplace=True)

    self.back_proj = nn.Linear(in_size, out_size * n_feats)

    self.mup = nn.MaxUnpool2d(2)

    self.ct1 = nn.ConvTranspose2d(n_feats, n_feats, 3)
    self.ct2 = nn.ConvTranspose2d(n_feats, n_feats, 3)
    self.ct3 = nn.ConvTranspose2d(n_feats, n_feats, 3)

    self.ct4 = nn.ConvTranspose2d(n_feats, n_feats, 3, padding=1)
    self.ct5 = nn.ConvTranspose2d(n_feats, n_feats, 5, padding=2)
    self.ct6 = nn.ConvTranspose2d(n_feats, 1, 7, padding=3)
  
  def forward(self, x, idx):
    bs, _, sl, _ = x.shape
    x = self.back_proj(x).reshape(bs, self.n_feats, sl, -1)

    out = self.mup(x, idx)

    out = self.relu(self.ct1(out))
    out = self.relu(self.ct2(out))
    out = self.relu(self.ct3(out))

    out = self.relu(self.ct4(out))
    out = self.relu(self.ct5(out))
    out = self.ct6(out)
    return out


class NaiveConvED(nn.Module):
  def __init__(self, encoder=None, decoder=None, n_feats=100, enc_in_size=400, enc_out_size=80):
    super().__init__()
    self.encoder = NaiveConvEncoder(n_feats=n_feats, in_size=enc_in_size, out_size=enc_out_size) if encoder is None else encoder
    self.decoder = NaiveConvDecoder(n_feats=n_feats, in_size=enc_out_size, out_size=self.encoder.in_size) if decoder is None else decoder
  
  def forward(self, x):  # x = [bs, seq_len, n_feats]
    out, idx = self.encoder(x.unsqueeze(1))
    return self.decoder(out, idx).squeeze(1)