import torch
from torch import nn
import torch.nn.functional as F

from .deformable_3d import DeformConv3d


class AffinityFeature(nn.Module):
    def __init__(self, win_h=3, win_w=3, dilation=1):
        super().__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0.0

    def forward(self, feature):
        # feature: [B, C, H, W]
        b, c, h, w = feature.size()
        feature = F.normalize(feature, dim=1, p=2)
        unfold_feature = nn.Unfold(
            kernel_size=(self.win_h, self.win_w),
            dilation=self.dilation,
            padding=self.dilation,
        )(feature)
        all_neighbor = unfold_feature.reshape(b, c, -1, h, w).transpose(1, 2)
        num = (self.win_h * self.win_w) // 2
        neighbor = torch.cat((all_neighbor[:, :num], all_neighbor[:, num + 1 :]), dim=1)
        feature = feature.unsqueeze(1)
        affinity = torch.sum(neighbor * feature, dim=2)
        affinity[affinity < self.cut] = self.cut
        return affinity


class CPA3D(nn.Module):
    def __init__(self, in_dim, dilations=(1, 2, 4)):
        super().__init__()
        self.dilations = dilations
        self.curr_convs = nn.ModuleList()
        self.hist_convs = nn.ModuleList()
        for d in dilations:
            pad = d
            self.curr_convs.append(
                nn.Sequential(
                    nn.Conv3d(in_dim, in_dim, 3, 1, padding=pad, dilation=d, bias=False),
                    nn.GroupNorm(1, in_dim),
                    nn.ReLU(inplace=True),
                )
            )
            self.hist_convs.append(
                nn.Sequential(
                    nn.Conv3d(in_dim, in_dim, 3, 1, padding=pad, dilation=d, bias=False),
                    nn.GroupNorm(1, in_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fuse = nn.Conv3d(len(dilations), 1, kernel_size=1, stride=1, padding=0)

    def forward(self, curr, hist):
        # curr/hist: [B, C, X, Y, Z]
        sims = []
        for cconv, hconv in zip(self.curr_convs, self.hist_convs):
            c = cconv(curr)
            h = hconv(hist)
            c = F.normalize(c, dim=1)
            h = F.normalize(h, dim=1)
            sim = (c * h).sum(dim=1, keepdim=True)  # [B,1,X,Y,Z]
            sims.append(sim)
        sims = torch.cat(sims, dim=1)  # [B, L, X, Y, Z]
        a = torch.sigmoid(self.fuse(sims))
        return a


class ADR3D(nn.Module):
    def __init__(self, in_channel=32, out_channel=None, mode="auto"):
        super().__init__()
        out_channel = out_channel or in_channel
        assert mode in ("auto", "deform", "plain")
        self.mode = mode

        if self.mode in ("auto", "deform"):
            self.deformable3 = DeformConv3d(in_channel, 8, kernel_size=3, stride=1, padding=1, bias=False)
            self.deformable5 = DeformConv3d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
            self.deformable7 = DeformConv3d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.deformable3 = None
            self.deformable5 = None
            self.deformable7 = None

        if self.mode in ("auto", "plain"):
            self.plain3 = nn.Conv3d(in_channel, 8, kernel_size=3, stride=1, padding=1, bias=False)
            self.plain5 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
            self.plain7 = nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.plain3 = None
            self.plain5 = None
            self.plain7 = None

        self.fuse = nn.Conv3d(8 * 3, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.contiguous().float()
        if self.mode == "plain":
            use_plain = True
        elif self.mode == "deform":
            use_plain = False
        else:
            # Use plain conv at 1/2 and full resolution; keep deformable at 1/4.
            use_plain = x.shape[2] >= 64 or x.shape[3] >= 64
        if use_plain:
            d3 = self.relu(self.plain3(x))
            d5 = self.relu(self.plain5(d3))
            d7 = self.relu(self.plain7(d5))
        else:
            d3 = self.relu(self.deformable3(x))
            d5 = self.relu(self.deformable5(d3))
            d7 = self.relu(self.deformable7(d5))
        out = self.fuse(torch.cat([d3, d5, d7], dim=1))
        return out


class FusionBlock3D(nn.Module):
    def __init__(self, channels, cpa_dilations=(1, 2, 4), adr_mode="auto", mask_blend_beta=0.7):
        super().__init__()
        self.cpa = CPA3D(channels, dilations=cpa_dilations)
        self.adr = ADR3D(channels * 2, out_channel=channels, mode=adr_mode)
        # For inv/occ voxels, softly blend toward history-refined features.
        # beta=1.0 means hard replacement with refined_hist.
        self.mask_blend_beta = float(mask_blend_beta)
        assert 0.0 <= self.mask_blend_beta <= 1.0

    def compute_affinity(self, curr, hist):
        # curr/hist: [B, C, X, Y, Z]
        a = self.cpa(curr, hist)
        refined_hist = self.adr(torch.cat([curr, hist], dim=1))
        fused_aff = a * refined_hist + (1.0 - a) * curr
        return fused_aff, refined_hist

    def forward(self, curr, hist, conf, inv_mask=None, occ_mask=None, precomputed=None):
        # curr/hist: [B, C, X, Y, Z], conf: [B, 1, X, Y, Z]
        if precomputed is None:
            fused_aff, refined_hist = self.compute_affinity(curr, hist)
        else:
            fused_aff, refined_hist = precomputed

        # step 1: affinity fusion (from CPA+ADR)
        # step 2: confidence gating
        fused = conf * fused_aff + (1.0 - conf) * curr

        blend_mask = None
        if inv_mask is not None:
            inv = inv_mask.to(dtype=torch.bool, device=fused.device)
            if inv.dim() == 3:
                inv = inv.unsqueeze(0).unsqueeze(0)
            blend_mask = inv
        if occ_mask is not None:
            occ = occ_mask.to(dtype=torch.bool, device=fused.device)
            if occ.dim() == 3:
                occ = occ.unsqueeze(0).unsqueeze(0)
            blend_mask = occ if blend_mask is None else (blend_mask | occ)

        if blend_mask is not None:
            beta = self.mask_blend_beta
            hist_prior = beta * refined_hist + (1.0 - beta) * curr
            fused = torch.where(blend_mask, hist_prior, fused)

        return fused, refined_hist
