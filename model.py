"""
CNN for CAMELS maps -> 6 parameters.

Two heads:
  'mse'    -> 6 outputs, the point estimate.  Start here.
  'moment' -> 12 outputs (6 means + 6 log-variances) trained with the two-moment
              loss, so the network also reports its own per-parameter error bar.
              This is what the CAMELS papers use.  Switch to it once 'mse' works.

    from model import ParamCNN, moment_loss
    net = ParamCNN(in_channels=1, head='mse')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PARAM = 6


class ConvBlock(nn.Module):
    """stride-2 downsample, then a stride-1 conv at the new resolution."""

    def __init__(self, cin, cout, slope=0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(slope, inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class ParamCNN(nn.Module):
    def __init__(self, in_channels=1, head="mse", widths=(32, 64, 128, 256, 256, 256),
                 dropout=0.2, n_param=N_PARAM):
        """
        widths : one entry per stride-2 block.  6 blocks take 256 -> 4 px.
                 Use 4 blocks (256->16) if you train on downsample=4 maps.
        """
        super().__init__()
        assert head in ("mse", "moment")
        self.head_kind = head
        self.n_param = n_param

        blocks, c = [], in_channels
        for w in widths:
            blocks.append(ConvBlock(c, w))
            c = w
        self.features = nn.Sequential(*blocks)

        # Global average + global std pooling.  The std channel matters here:
        # the *variance* of the map carries sigma_8 information that a plain
        # mean-pool would throw away.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2 * c, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_param * (2 if head == "moment" else 1)),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.features(x)                       # (B, C, h, w)
        mean = f.mean(dim=(2, 3))
        std = f.std(dim=(2, 3))
        return self.head(torch.cat([mean, std], dim=1))

    # ---- convenience: always get (mu, sigma) whatever the head -----------
    def predict(self, x):
        out = self.forward(x)
        if self.head_kind == "moment":
            mu, logvar = out[:, :self.n_param], out[:, self.n_param:]
            return mu, torch.exp(0.5 * logvar.clamp(-20, 20))
        return out, torch.full_like(out, float("nan"))


# ---------------------------------------------------------------------------
# Losses.  Both operate on targets already scaled to [0, 1].
# ---------------------------------------------------------------------------
def mse_loss(out, y):
    return F.mse_loss(out, y)


def moment_loss(out, y, eps=1e-12):
    """
    Two-moment / likelihood-free loss (Jeffrey & Wandelt; used by CAMELS).

    Minimising term 1 drives mu -> E[theta | map].
    Minimising term 2 drives sigma^2 -> Var[theta | map].
    Both are taken *per parameter over the batch*, hence the .mean(0) before
    the log -- this needs a batch of a few tens to be well behaved.
    """
    n = out.shape[1] // 2
    mu, logvar = out[:, :n], out[:, n:]
    var = torch.exp(logvar.clamp(-20, 20))
    d2 = (y - mu) ** 2
    term1 = torch.log(d2.mean(dim=0) + eps).sum()
    term2 = torch.log(((d2 - var) ** 2).mean(dim=0) + eps).sum()
    return term1 + term2


def get_loss(head):
    return mse_loss if head == "mse" else moment_loss


if __name__ == "__main__":
    for head in ("mse", "moment"):
        net = ParamCNN(in_channels=1, head=head)
        x = torch.randn(4, 1, 256, 256)
        y = torch.rand(4, 6)
        out = net(x)
        n = sum(p.numel() for p in net.parameters())
        print(f"{head:7s} out={tuple(out.shape)} params={n/1e6:.2f}M "
              f"loss={get_loss(head)(out, y).item():.4f}")
        mu, sig = net.predict(x)
        print(f"        mu={tuple(mu.shape)} sigma={tuple(sig.shape)}")
    # 4-block variant for 64x64 inputs
    net = ParamCNN(in_channels=1, head="mse", widths=(32, 64, 128, 256))
    print("64px  out=", tuple(net(torch.randn(2, 1, 64, 64)).shape))
