import torch
import torch.nn as nn
import torch.nn.functional as F

from ttvidt.utils import compile_wrapper


class TemporalPatch(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        patch_size=2,
        overlap=1,
        init_gamma=1.1,
    ):
        super(TemporalPatch, self).__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.input_channels = input_channels
        self.output_channels = output_channels
        if input_channels != output_channels:
            self.fproj = nn.Conv1d(input_channels, output_channels, 1, 1, 0)
        else:
            self.fproj = nn.Identity()
        self.tproj = nn.Conv1d(
            output_channels, output_channels, patch_size + overlap, stride=patch_size
        )
        self.init_gamma = init_gamma
        self.init_weight()

    def init_weight(self):
        if isinstance(self.fproj, nn.Conv1d):
            nn.init.constant_(self.fproj.bias, 0)
            nn.init.normal_(self.fproj.weight, std=1 / self.input_channels**0.5)

        nn.init.constant_(self.tproj.bias, 0)
        tproj_patch = self.tproj.weight.size(-1)
        I = torch.eye(self.tproj.weight.size(0)).to(self.tproj.weight)[..., None]
        weight = I.repeat(1, 1, tproj_patch) * self.init_gamma
        scale = [self.init_gamma**i for i in range(tproj_patch)]
        scale = torch.tensor(scale).to(self.tproj.weight) / sum(scale)
        self.tproj.weight.data.copy_(weight * scale[None, None, :])

    @compile_wrapper
    def forward(self, x):
        """
        x: [B, T, L, C]
        """
        b, t, l, c = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b * l, c, t)

        x = self.fproj(x)
        first_frame = x[..., 0:1]
        x = self.tproj(x)
        x = torch.concat([first_frame, x], dim=-1)

        _, oc, ot = x.shape
        x = x.view(b, l, oc, ot).permute(0, 3, 1, 2).contiguous()
        return x


if __name__ == "__main__":
    x = torch.randn(2, 1 + 15 * 8, 3, 224, 224)
    m = TemporalPatch(3, 4, 8, 1)
    y = m(x)  # [B, 1 + (T-1)//p, 4, H, W]
    print(y.shape)
