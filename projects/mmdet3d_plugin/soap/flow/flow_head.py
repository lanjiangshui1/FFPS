import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer


@HEADS.register_module()
class FlowHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        soft_weights=False,
        conv_cfg=dict(type="Conv3d", bias=False),
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
    ):
        super().__init__()
        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level

        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i]
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True),
            )
            self.occ_convs.append(occ_conv)

        self.occ_pred_conv = nn.Sequential(
            build_conv_layer(conv_cfg, in_channels=mid_channel, out_channels=mid_channel // 2, kernel_size=1, stride=1, padding=0),
            build_norm_layer(norm_cfg, mid_channel // 2)[1],
            nn.ReLU(inplace=True),
        )
        self.last_conv = build_conv_layer(conv_cfg, in_channels=mid_channel // 2, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        self.soft_weights = soft_weights
        self.num_point_sampling_feat = self.num_level
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, out_channels=soft_in_channel // 2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel // 2, out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0),
            )

    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            output_occs.append(occ_conv(feats))
        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1], device=output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, h, w, d = output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[h, w, d], mode="trilinear", align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        output["out_voxel_feats"] = [out_voxel_feats]

        out_voxel = self.occ_pred_conv(out_voxel_feats)
        out_voxel = self.last_conv(out_voxel)
        output["occ"] = [out_voxel]
        return output

    def forward(self, voxel_feats, **kwargs):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        output = self.forward_coarse_voxel(voxel_feats)
        res = {"output_voxels": output["occ"]}
        return res
