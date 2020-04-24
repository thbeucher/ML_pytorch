import torch.nn as nn


class NaiveConvEncoder(nn.Module):
  def __init__(self, n_feats=100):
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
    return out, idx


class NaiveConvDecoder(nn.Module):
  def __init__(self, n_feats=100):
    super().__init__()
    self.relu = nn.ReLU(inplace=True)

    self.mup = nn.MaxUnpool2d(2)

    self.ct1 = nn.ConvTranspose2d(n_feats, n_feats, 3)
    self.ct2 = nn.ConvTranspose2d(n_feats, n_feats, 3)
    self.ct3 = nn.ConvTranspose2d(n_feats, n_feats, 3)

    self.ct4 = nn.ConvTranspose2d(n_feats, n_feats, 3, padding=1)
    self.ct5 = nn.ConvTranspose2d(n_feats, n_feats, 5, padding=2)
    self.ct6 = nn.ConvTranspose2d(n_feats, 1, 7, padding=3)
  
  def forward(self, x, idx):
    out = self.mup(x, idx)

    out = self.relu(self.ct1(out))
    out = self.relu(self.ct2(out))
    out = self.relu(self.ct3(out))

    out = self.relu(self.ct4(out))
    out = self.relu(self.ct5(out))
    out = self.relu(self.ct6(out))
    return out


class NaiveConvED(nn.Module):
  def __init__(self, encoder=None, decoder=None, n_feats=100):
    super().__init__()
    self.encoder = NaiveConvEncoder(n_feats=n_feats) if encoder is None else encoder
    self.decoder = NaiveConvDecoder(n_feats=n_feats) if decoder is None else decoder
  
  def forward(self, x):  # x = [bs, seq_len, n_feats]
    out, idx = self.encoder(x.unsqueeze(1))
    return self.decoder(out, idx).squeeze(1)