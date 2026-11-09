import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import NECKS


@NECKS.register_module()
class FPN3D(BaseModule):
    def __init__(
        self,
        in_channels=[80, 160, 320, 640],
        out_channels=256,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        conv_cfg=dict(type="Conv3d"),
        act_cfg=dict(type="ReLU"),
        with_cp=False,
        upsample_cfg=dict(mode="trilinear"),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        self.num_out = len(self.in_channels)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_out):
            l_conv = nn.Sequential(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=False,
                    inplace=True,
                ),
            )
            fpn_conv = nn.Sequential(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=False,
                    inplace=True,
                ),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            lateral_i = lateral_conv(inputs[i])
            laterals.append(lateral_i)

        for i in range(self.num_out - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, align_corners=False, **self.upsample_cfg)

        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            outs.append(fpn_conv(laterals[i]))
        return outs
