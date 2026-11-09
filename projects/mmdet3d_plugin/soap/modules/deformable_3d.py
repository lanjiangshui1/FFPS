import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class DeformConv3d(nn.Module):
    """Pure PyTorch 3D deformable convolution (adapted from HTCL)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        n = kernel_size ** 3
        self.stride = stride
        self.padding = padding
        self.zero_padding = nn.ConstantPad3d(padding, 0)
        self.conv_kernel = nn.Conv3d(in_channels * n, out_channels, kernel_size=1, bias=bias)
        self.offset_conv_kernel = nn.Conv3d(in_channels, n * 3, kernel_size=kernel_size, padding=padding, bias=bias)
        self.mode = "deformable"

    def deformable_mode(self, on=True):
        self.mode = "deformable" if on else "regular"

    def forward(self, x):
        if self.mode == "deformable":
            offset = self.offset_conv_kernel(x)
        else:
            b, c, h, w, d = x.size()
            offset = torch.zeros(b, 3 * self.kernel_size ** 3, h, w, d, device=x.device, dtype=x.dtype)

        dtype = offset.data.type()
        ks = self.kernel_size
        n = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)

        p = self._get_p(offset, dtype)
        p = p[:, :, :: self.stride, :: self.stride, :: self.stride]
        p = p.contiguous().permute(0, 2, 3, 4, 1)

        q_sss = Variable(p.data, requires_grad=False).floor()
        q_lll = q_sss + 1

        q_sss = torch.cat(
            [
                torch.clamp(q_sss[..., :n], 0, x.size(2) - 1),
                torch.clamp(q_sss[..., n : 2 * n], 0, x.size(3) - 1),
                torch.clamp(q_sss[..., 2 * n :], 0, x.size(4) - 1),
            ],
            dim=-1,
        ).long()
        q_lll = torch.cat(
            [
                torch.clamp(q_lll[..., :n], 0, x.size(2) - 1),
                torch.clamp(q_lll[..., n : 2 * n], 0, x.size(3) - 1),
                torch.clamp(q_lll[..., 2 * n :], 0, x.size(4) - 1),
            ],
            dim=-1,
        ).long()
        q_ssl = torch.cat([q_sss[..., :n], q_sss[..., n : 2 * n], q_lll[..., 2 * n :]], -1)
        q_sls = torch.cat([q_sss[..., :n], q_lll[..., n : 2 * n], q_sss[..., 2 * n :]], -1)
        q_sll = torch.cat([q_sss[..., :n], q_lll[..., n : 2 * n], q_lll[..., 2 * n :]], -1)
        q_lss = torch.cat([q_lll[..., :n], q_sss[..., n : 2 * n], q_sss[..., 2 * n :]], -1)
        q_lsl = torch.cat([q_lll[..., :n], q_sss[..., n : 2 * n], q_lll[..., 2 * n :]], -1)
        q_lls = torch.cat([q_lll[..., :n], q_lll[..., n : 2 * n], q_sss[..., 2 * n :]], -1)

        mask = torch.cat(
            [
                p[..., :n].lt(self.padding) + p[..., :n].gt(x.size(2) - 1 - self.padding),
                p[..., n : 2 * n].lt(self.padding) + p[..., n : 2 * n].gt(x.size(3) - 1 - self.padding),
                p[..., 2 * n :].lt(self.padding) + p[..., 2 * n :].gt(x.size(4) - 1 - self.padding),
            ],
            dim=-1,
        ).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))
        p = p * (1 - mask) + floor_p * mask

        p = torch.cat(
            [
                torch.clamp(p[..., :n], 0, x.size(2) - 1),
                torch.clamp(p[..., n : 2 * n], 0, x.size(3) - 1),
                torch.clamp(p[..., 2 * n :], 0, x.size(4) - 1),
            ],
            dim=-1,
        )

        g_sss = (1 + (q_sss[..., :n].type_as(p) - p[..., :n])) * (
            1 + (q_sss[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 + (q_sss[..., 2 * n :].type_as(p) - p[..., 2 * n :]))
        g_lll = (1 - (q_lll[..., :n].type_as(p) - p[..., :n])) * (
            1 - (q_lll[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 - (q_lll[..., 2 * n :].type_as(p) - p[..., 2 * n :]))
        g_ssl = (1 + (q_ssl[..., :n].type_as(p) - p[..., :n])) * (
            1 + (q_ssl[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 - (q_ssl[..., 2 * n :].type_as(p) - p[..., 2 * n :]))
        g_sls = (1 + (q_sls[..., :n].type_as(p) - p[..., :n])) * (
            1 - (q_sls[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 + (q_sls[..., 2 * n :].type_as(p) - p[..., 2 * n :]))
        g_sll = (1 + (q_sll[..., :n].type_as(p) - p[..., :n])) * (
            1 - (q_sll[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 - (q_sll[..., 2 * n :].type_as(p) - p[..., 2 * n :]))
        g_lss = (1 - (q_lss[..., :n].type_as(p) - p[..., :n])) * (
            1 + (q_lss[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 + (q_lss[..., 2 * n :].type_as(p) - p[..., 2 * n :]))
        g_lsl = (1 - (q_lsl[..., :n].type_as(p) - p[..., :n])) * (
            1 + (q_lsl[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 - (q_lsl[..., 2 * n :].type_as(p) - p[..., 2 * n :]))
        g_lls = (1 - (q_lls[..., :n].type_as(p) - p[..., :n])) * (
            1 - (q_lls[..., n : 2 * n].type_as(p) - p[..., n : 2 * n])
        ) * (1 + (q_lls[..., 2 * n :].type_as(p) - p[..., 2 * n :]))

        x_q_sss = self._get_x_q(x, q_sss, n)
        x_q_lll = self._get_x_q(x, q_lll, n)
        x_q_ssl = self._get_x_q(x, q_ssl, n)
        x_q_sls = self._get_x_q(x, q_sls, n)
        x_q_sll = self._get_x_q(x, q_sll, n)
        x_q_lss = self._get_x_q(x, q_lss, n)
        x_q_lsl = self._get_x_q(x, q_lsl, n)
        x_q_lls = self._get_x_q(x, q_lls, n)

        x_offset = (
            g_sss.unsqueeze(dim=1) * x_q_sss
            + g_lll.unsqueeze(dim=1) * x_q_lll
            + g_ssl.unsqueeze(dim=1) * x_q_ssl
            + g_sls.unsqueeze(dim=1) * x_q_sls
            + g_sll.unsqueeze(dim=1) * x_q_sll
            + g_lss.unsqueeze(dim=1) * x_q_lss
            + g_lsl.unsqueeze(dim=1) * x_q_lsl
            + g_lls.unsqueeze(dim=1) * x_q_lls
        )

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)
        return out

    def _get_p_n(self, n, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing="ij",
        )
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * n, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)
        return p_n

    @staticmethod
    def _get_p_0(h, w, d, n, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing="ij")
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(n, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(n, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(n, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        n, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)
        p_n = self._get_p_n(n, dtype).to(offset)
        p_0 = self._get_p_0(h, w, d, n, dtype).to(offset)
        return p_0 + p_n + offset

    def _get_x_q(self, x, q, n):
        b, h, w, d, _ = q.size()
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :n] * padded_w * padded_d + q[..., n : 2 * n] * padded_d + q[..., 2 * n :]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, n)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, n = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        x_offset = x_offset.contiguous().view(b, c * n, h, w, d)
        return x_offset
