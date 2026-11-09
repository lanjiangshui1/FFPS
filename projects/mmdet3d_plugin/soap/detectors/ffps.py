import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmcv.runner import force_fp32

from .bevdepth import BEVDepth
from projects.mmdet3d_plugin.soap.modules.cpa_adr import FusionBlock3D
from projects.mmdet3d_plugin.soap.utils.ssc_loss import CE_ssc_loss, sem_scal_loss, geo_scal_loss
from projects.mmdet3d_plugin.utils.semkitti import get_class_frequencies


@DETECTORS.register_module()
class FFPS(BEVDepth):
    def __init__(
        self,
        n_classes,
        flow_encoder_backbone=None,
        flow_predictor=None,
        flow_encoder_neck=None,
        flow_head=None,
        flow_conf_channels=1,
        aux_loss_weight=1.0,
        loss_flow_aux_weight=0.5,
        loss_flow_aux_min_weight=0.2,
        loss_flow_rigid_weight=0.02,
        loss_align_weight=0.05,
        loss_conf_weight=0.05,
        loss_conf_entropy_weight=0.01,
        loss_conf_half_weight=0.05,
        loss_conf_full_weight=0.05,
        loss_conf_final_weight=0.05,
        loss_conf_half_entropy_weight=0.01,
        loss_conf_full_entropy_weight=0.01,
        loss_conf_final_entropy_weight=0.01,
        flow_aux_schedule="linear",
        flow_aux_warmup_ratio=0.2,
        flow_aux_mid_ratio=0.7,
        flow_aux_mid_scale=1.1,
        enable_aux_ce_reweight=False,
        aux_ce_reweight_power=1.0,
        aux_ce_rare_boost=None,
        class_frequency_source=None,
        loss_weight_cfg=None,
        empty_idx=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.aux_loss_weight = aux_loss_weight
        self.loss_flow_aux_weight = float(loss_flow_aux_weight)
        self.loss_flow_aux_min_weight = float(loss_flow_aux_min_weight)
        self.loss_flow_rigid_weight = float(loss_flow_rigid_weight)
        self.loss_align_weight = loss_align_weight
        self.loss_conf_weight = loss_conf_weight
        self.loss_conf_entropy_weight = loss_conf_entropy_weight
        self.loss_conf_half_weight = loss_conf_half_weight
        self.loss_conf_full_weight = loss_conf_full_weight
        self.loss_conf_final_weight = loss_conf_final_weight
        self.loss_conf_half_entropy_weight = loss_conf_half_entropy_weight
        self.loss_conf_full_entropy_weight = loss_conf_full_entropy_weight
        self.loss_conf_final_entropy_weight = loss_conf_final_entropy_weight
        self.flow_aux_schedule = str(flow_aux_schedule)
        self.flow_aux_warmup_ratio = float(flow_aux_warmup_ratio)
        self.flow_aux_mid_ratio = float(flow_aux_mid_ratio)
        self.flow_aux_mid_scale = float(flow_aux_mid_scale)
        self.enable_aux_ce_reweight = bool(enable_aux_ce_reweight)
        self.aux_ce_reweight_power = float(aux_ce_reweight_power)
        self.aux_ce_rare_boost = aux_ce_rare_boost
        self.class_frequency_source = class_frequency_source

        self.flow_encoder_backbone = builder.build_backbone(flow_encoder_backbone)
        self.flow_predictor = builder.build_backbone(flow_predictor)
        self.flow_encoder_neck = builder.build_neck(flow_encoder_neck)
        self.flow_head = builder.build_head(flow_head)

        flow_conf_in = flow_encoder_neck.get("out_channels", None)
        if isinstance(flow_conf_in, list):
            flow_conf_in = flow_conf_in[-1]
        self.flow_conf_head = nn.Sequential(
            nn.Conv3d(flow_conf_in * 2, flow_conf_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        mask_blend_beta = 0.55
        self.fusion_1_4 = FusionBlock3D(
            flow_conf_in, cpa_dilations=(1, 2, 4), adr_mode="deform", mask_blend_beta=mask_blend_beta
        )
        self.fusion_1_2 = FusionBlock3D(
            flow_conf_in, cpa_dilations=(1, 2), adr_mode="plain", mask_blend_beta=mask_blend_beta
        )
        self.fusion_1_1 = FusionBlock3D(
            flow_conf_in, cpa_dilations=(1,), adr_mode="plain", mask_blend_beta=mask_blend_beta
        )
        self.fusion_1_0 = FusionBlock3D(
            flow_conf_in, cpa_dilations=(1,), adr_mode="plain", mask_blend_beta=mask_blend_beta
        )

        self.conf_half_head = nn.Sequential(
            nn.Conv3d(flow_conf_in * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.conf_full_head = nn.Sequential(
            nn.Conv3d(flow_conf_in * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.conf_final_head = nn.Sequential(
            nn.Conv3d(flow_conf_in * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.aux_head = nn.Conv3d(flow_conf_in, n_classes, kernel_size=1, stride=1, padding=0)

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        self.empty_idx = empty_idx
        self.class_weights = None
        if self.enable_aux_ce_reweight:
            class_frequencies = get_class_frequencies(self.class_frequency_source, int(n_classes))
            aux_class_weights = 1.0 / np.log(class_frequencies.astype(np.float64))
            aux_class_weights = aux_class_weights / max(aux_class_weights[0], 1e-6)
            aux_class_weights = np.power(aux_class_weights, self.aux_ce_reweight_power)
            if self.aux_ce_rare_boost is not None:
                boost = np.asarray(self.aux_ce_rare_boost, dtype=np.float64)
                if boost.shape[0] != int(n_classes):
                    raise ValueError(
                        f"aux_ce_rare_boost length ({boost.shape[0]}) must equal n_classes ({n_classes})."
                    )
                aux_class_weights = aux_class_weights * boost
            self.class_weights = torch.tensor(aux_class_weights, dtype=torch.float32)

        if isinstance(kwargs["img_neck"]["out_channels"], list):
            feat_c = kwargs["img_neck"]["out_channels"][0]
        else:
            feat_c = kwargs["img_neck"]["out_channels"]
        self.sem_dic_scores = nn.Parameter(torch.zeros(n_classes, 10), requires_grad=False)
        self.sem_dic = nn.Parameter(torch.zeros(n_classes, 10, feat_c), requires_grad=False)
        self.dic_embed = nn.Embedding(n_classes, feat_c)
        nn.init.xavier_uniform_(self.dic_embed.weight)
        self.update_for_infer = False

    def image_encoder(self, img):
        imgs = img
        b, n, c, imh, imw = imgs.shape
        imgs = imgs.view(b * n, c, imh, imw)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    @force_fp32()
    def bev_encoder(self, mlvl_voxel_features, geo_inputs, mlvl_masks):
        x = self.img_bev_encoder_backbone(mlvl_voxel_features, geo_inputs, mlvl_masks)
        x = self.img_bev_encoder_neck(x)
        return x

    @force_fp32()
    def extract_img_feat(self, img, img_metas):
        nframes = img[0].size(1)
        curr_img, prev_img = torch.split(img[0], [1, nframes - 1], dim=1)

        curr_mlvl_feats = self.image_encoder(curr_img)
        # Keep gradients for history while reducing activation memory.
        if self.training:
            prev_img_ckpt = prev_img.detach()
            prev_img_ckpt.requires_grad_(True)
            prev_mlvl_feats = checkpoint(self.image_encoder, prev_img_ckpt)
        else:
            prev_mlvl_feats = self.image_encoder(prev_img)

        rots, trans, rel_rots, rel_trans, intrins, post_rots, post_trans, bda = img[1:9]
        geo_inputs_all = [rots, trans, intrins, post_rots, post_trans]
        curr_geo_inputs = [g[:, :1] for g in geo_inputs_all] + [bda]
        prev_geo_inputs = [g[:, 1:] for g in geo_inputs_all] + [bda]
        curr_mlp_input = self.img_view_transformer.get_mlp_input(*curr_geo_inputs)
        prev_mlp_input = self.img_view_transformer.get_mlp_input(*prev_geo_inputs)
        curr_geo_inputs = curr_geo_inputs[:2] + [rel_rots, rel_trans] + curr_geo_inputs[2:] + [curr_mlp_input]
        prev_geo_inputs = prev_geo_inputs[:2] + [rel_rots, rel_trans] + prev_geo_inputs[2:] + [prev_mlp_input]

        mlvl_curr_feats, mlvl_prev_feats, mlvl_masks, mlvl_inv_masks, mlvl_occ_masks = self.img_view_transformer(
            img[9], curr_mlvl_feats, prev_mlvl_feats, curr_geo_inputs, prev_geo_inputs
        )
        return mlvl_curr_feats, mlvl_prev_feats, mlvl_masks, mlvl_inv_masks, mlvl_occ_masks, curr_geo_inputs

    def flow_encoder(self, x):
        b, t, c, xh, yw, zd = x.shape
        x = x.reshape(b, t * c, xh, yw, zd)
        x = self.flow_encoder_backbone(x)
        x = self.flow_predictor(x)
        x = self.flow_encoder_neck(x)
        return x

    def _select_flow_head_feats(self, flow_feats):
        num_level = getattr(self.flow_head, "num_level", 1)
        if num_level == len(flow_feats):
            return flow_feats
        if num_level == 1:
            return [flow_feats[-1]]
        assert num_level <= len(flow_feats), "flow_head.num_level exceeds available feature levels"
        return flow_feats[-num_level:]

    def _get_aux_loss_weight(self):
        runner = getattr(self, "runner", None)
        if runner is None:
            return self.aux_loss_weight
        max_epochs = getattr(runner, "max_epochs", None)
        epoch = getattr(runner, "epoch", None)
        if max_epochs is None or epoch is None or max_epochs <= 0:
            return self.aux_loss_weight
        progress = float(epoch) / float(max_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return max(0.2, self.aux_loss_weight * (1.0 - 0.8 * progress))

    def _get_flow_aux_weight(self):
        runner = getattr(self, "runner", None)
        if runner is None:
            return self.loss_flow_aux_weight
        max_epochs = getattr(runner, "max_epochs", None)
        epoch = getattr(runner, "epoch", None)
        if max_epochs is None or epoch is None or max_epochs <= 0:
            return self.loss_flow_aux_weight
        progress = float(epoch) / float(max_epochs)
        progress = min(max(progress, 0.0), 1.0)
        if self.flow_aux_schedule == "piecewise":
            warm = min(max(self.flow_aux_warmup_ratio, 0.0), 0.95)
            mid = min(max(self.flow_aux_mid_ratio, warm + 1e-3), 0.99)
            base = self.loss_flow_aux_weight
            min_w = self.loss_flow_aux_min_weight
            mid_w = max(min_w, base * self.flow_aux_mid_scale)
            if progress < warm:
                warm_scale = 0.8 + 0.2 * (progress / max(warm, 1e-6))
                return max(min_w, base * warm_scale)
            if progress < mid:
                return max(min_w, mid_w)
            tail = (progress - mid) / max(1.0 - mid, 1e-6)
            return max(min_w, (1.0 - tail) * mid_w + tail * min_w)
        return max(
            self.loss_flow_aux_min_weight,
            self.loss_flow_aux_weight * (1.0 - 0.8 * progress),
        )

    def _build_rigid_flow_prior(self, flow_logits, geo_inputs, img_metas):
        if geo_inputs is None or img_metas is None or len(geo_inputs) <= 3:
            return None
        rel_trans = geo_inputs[3]
        if rel_trans is None or rel_trans.dim() < 3 or rel_trans.shape[1] <= 1:
            return None
        if not isinstance(img_metas, (list, tuple)) or len(img_metas) == 0 or not isinstance(img_metas[0], dict):
            return None
        pc_range = img_metas[0].get("pc_range", None)
        occ_size = img_metas[0].get("occ_size", None)
        if pc_range is None or occ_size is None:
            return None

        pc_range = torch.as_tensor(pc_range, device=flow_logits.device, dtype=flow_logits.dtype)
        occ_size = torch.as_tensor(occ_size, device=flow_logits.device, dtype=flow_logits.dtype)
        if pc_range.numel() != 6 or occ_size.numel() != 3:
            return None

        voxel_size = (pc_range[3:] - pc_range[:3]) / occ_size
        voxel_size = torch.clamp(voxel_size, min=1e-6)
        trans_prev = rel_trans[:, 1:, :].mean(dim=1).to(dtype=flow_logits.dtype, device=flow_logits.device)

        rigid = torch.zeros_like(flow_logits)
        rigid[:, 0] = (-trans_prev[:, 0] / voxel_size[0]).view(-1, 1, 1, 1)
        rigid[:, 1] = (-trans_prev[:, 1] / voxel_size[1]).view(-1, 1, 1, 1)
        rigid[:, 2] = (-trans_prev[:, 2] / voxel_size[2]).view(-1, 1, 1, 1)
        return rigid

    def warp_features(self, x, flow):
        # x: [B, C, X, Y, Z], flow: [B, 3, X, Y, Z]
        b, _, xs, ys, zs = x.shape
        x_in = x.permute(0, 1, 4, 3, 2)
        flow = flow.permute(0, 1, 4, 3, 2)

        grid_z = torch.linspace(-1, 1, zs, device=x.device, dtype=x.dtype)
        grid_y = torch.linspace(-1, 1, ys, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(-1, 1, xs, device=x.device, dtype=x.dtype)
        zz, yy, xx = torch.meshgrid(grid_z, grid_y, grid_x, indexing="ij")
        grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1, 1)

        flow_norm = torch.stack(
            [
                2.0 * flow[:, 0] / max(xs - 1, 1),
                2.0 * flow[:, 1] / max(ys - 1, 1),
                2.0 * flow[:, 2] / max(zs - 1, 1),
            ],
            dim=1,
        )
        grid = grid + flow_norm.permute(0, 2, 3, 4, 1)
        out = F.grid_sample(x_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        out = out.permute(0, 1, 4, 3, 2)
        return out

    def _flow_fuse_multiscale(
        self,
        mlvl_curr_feats,
        mlvl_prev_feats,
        mlvl_inv_masks,
        mlvl_occ_masks,
        gt_occ=None,
        geo_inputs=None,
        img_metas=None,
    ):
        assert len(mlvl_curr_feats) >= 4, "FFPS now expects 4-scale voxel features."

        # scale mapping: index 3 -> coarsest ... index 0 -> finest
        curr_s3 = mlvl_curr_feats[3]
        prev_s3 = mlvl_prev_feats[3]
        inv_s3 = mlvl_inv_masks[3]
        occ_s3 = mlvl_occ_masks[3]

        seq = torch.stack([prev_s3, curr_s3], dim=1)
        flow_feats = self.flow_encoder(seq)
        flow_logits = self.flow_head(self._select_flow_head_feats(flow_feats))["output_voxels"][0]
        if flow_logits.shape[-3:] != prev_s3.shape[-3:]:
            flow_logits = F.interpolate(
                flow_logits, size=prev_s3.shape[-3:], mode="trilinear", align_corners=False
            )
        hist_warp = self.warp_features(prev_s3, flow_logits)
        conf_s3 = self.flow_conf_head(torch.cat([curr_s3, hist_warp], dim=1))
        if conf_s3.shape[-3:] != curr_s3.shape[-3:]:
            conf_s3 = F.interpolate(conf_s3, size=curr_s3.shape[-3:], mode="trilinear", align_corners=False)

        # Coarsest scale: keep confidence only on inv/occ regions.
        inv_occ_s3 = (inv_s3 | occ_s3).unsqueeze(0).unsqueeze(0).to(conf_s3.dtype)
        conf_s3_fuse = conf_s3 * inv_occ_s3
        fused_s3, _ = self.fusion_1_4(curr_s3, hist_warp, conf_s3_fuse, inv_mask=inv_s3, occ_mask=occ_s3)

        aux_losses = {}
        loss_align = None
        loss_conf = None
        loss_conf_entropy = None
        loss_conf_half = None
        loss_conf_full = None
        loss_conf_final = None
        loss_conf_half_entropy = None
        loss_conf_full_entropy = None
        loss_conf_final_entropy = None
        loss_flow_rigid = None

        eps = 1e-6
        if gt_occ is not None:
            aux_logits = self.aux_head(fused_s3)
            aux_target = self._resize_gt(gt_occ, fused_s3.shape[-3:])
            aux_losses = self._loss_ssc(aux_logits, aux_target, prefix="aux_")

            valid_s3 = (~inv_s3) & (~occ_s3)
            valid_s3 = valid_s3.unsqueeze(0).unsqueeze(0).to(conf_s3.dtype)
            inv_s3_mask = inv_s3.unsqueeze(0).unsqueeze(0).to(conf_s3.dtype)
            occ_s3_mask = occ_s3.unsqueeze(0).unsqueeze(0).to(conf_s3.dtype)
            # Robust alignment: vis-dominant supervision with a weak occ regularizer.
            align_map = torch.sqrt((hist_warp - curr_s3).pow(2) + 1e-4).mean(dim=1, keepdim=True)
            align_weights = 1.0 * valid_s3 + 0.15 * occ_s3_mask
            loss_align = (align_map * align_weights).sum() / (align_weights.sum() + 1e-6)

            rigid_prior = self._build_rigid_flow_prior(flow_logits, geo_inputs, img_metas)
            if rigid_prior is not None:
                rigid_map = torch.sqrt((flow_logits - rigid_prior).pow(2) + 1e-4).mean(dim=1, keepdim=True)
                # Rigid prior should mainly regularize static voxels.
                flow_mag = torch.sqrt(flow_logits.pow(2).sum(dim=1, keepdim=True) + 1e-6)
                static_mask = (flow_mag < 1.0).to(valid_s3.dtype)
                rigid_weights = valid_s3 * static_mask
                if rigid_weights.sum() <= 1e-6:
                    rigid_weights = valid_s3
                loss_flow_rigid = (rigid_map * rigid_weights).sum() / (rigid_weights.sum() + 1e-6)

            # Soft confidence targets derived from alignment quality.
            align_det = align_map.detach()
            tau = ((align_det * inv_occ_s3).sum() / (inv_occ_s3.sum() + 1e-6)).clamp(min=1e-4)
            quality = torch.exp(-align_det / tau)
            conf_target_inv = 0.2 + 0.8 * quality
            conf_target_occ = 0.1 + 0.6 * quality
            conf_target = conf_target_inv * inv_s3_mask + conf_target_occ * occ_s3_mask
            conf_target = conf_target.clamp(min=0.05, max=0.99)
            conf_loss_map = F.binary_cross_entropy(conf_s3, conf_target, reduction="none")
            loss_conf = (conf_loss_map * inv_occ_s3).sum() / (inv_occ_s3.sum() + 1e-6)

            conf_clamped = conf_s3.clamp(min=eps, max=1.0 - eps)
            conf_entropy = -(conf_clamped * torch.log(conf_clamped) + (1.0 - conf_clamped) * torch.log(1.0 - conf_clamped))
            loss_conf_entropy = ((0.6931 - conf_entropy) * inv_occ_s3).sum() / (inv_occ_s3.sum() + 1e-6)

            if self.training:
                try:
                    occ_mask_dbg = occ_s3.unsqueeze(0).unsqueeze(0)
                    inv_mask_dbg = inv_s3.unsqueeze(0).unsqueeze(0)
                    occinv_mask_dbg = inv_occ_s3 > 0
                    conf_occinv_mean = conf_s3[occinv_mask_dbg].mean() if occinv_mask_dbg.any() else conf_s3.new_tensor(0.0)
                    conf_occ_mean = conf_s3[occ_mask_dbg].mean() if occ_mask_dbg.any() else conf_s3.new_tensor(0.0)
                    conf_inv_mean = conf_s3[inv_mask_dbg].mean() if inv_mask_dbg.any() else conf_s3.new_tensor(0.0)
                    msg = (
                        f"[debug] loss_align={loss_align.item():.6f}, "
                        f"valid_sum={valid_s3.sum().item():.6f}, "
                        f"inv_occ_sum={inv_occ_s3.sum().item():.6f}, "
                        f"conf_occinv_mean={conf_occinv_mean.item():.6f}, "
                        f"conf_occ_mean={conf_occ_mean.item():.6f}, "
                        f"conf_inv_mean={conf_inv_mean.item():.6f}"
                    )
                    if hasattr(self, "logger") and self.logger is not None:
                        self.logger.info(msg)
                    else:
                        print(msg)
                except Exception:
                    pass

        curr_s2 = mlvl_curr_feats[2]
        fused_s3_up = F.interpolate(fused_s3, size=curr_s2.shape[-3:], mode="trilinear", align_corners=False)
        fused_aff_s2, refined_hist_s2 = self.fusion_1_2.compute_affinity(curr_s2, fused_s3_up)
        conf_s2 = self.conf_half_head(torch.cat([curr_s2, fused_aff_s2], dim=1))
        inv_s2 = self._resize_mask(inv_s3, curr_s2.shape[-3:])
        occ_s2 = self._resize_mask(occ_s3, curr_s2.shape[-3:])
        inv_occ_s2 = (inv_s2 | occ_s2).unsqueeze(0).unsqueeze(0).to(conf_s2.dtype)
        conf_s2_fuse = conf_s2 * inv_occ_s2
        conf_s2_target = inv_occ_s2
        if gt_occ is not None:
            conf_s2_loss_map = F.binary_cross_entropy(conf_s2, conf_s2_target, reduction="none")
            loss_conf_half = (conf_s2_loss_map * inv_occ_s2).sum() / (inv_occ_s2.sum() + 1e-6)
            conf_s2_clamped = conf_s2.clamp(min=eps, max=1.0 - eps)
            conf_s2_entropy = -(conf_s2_clamped * torch.log(conf_s2_clamped) + (1.0 - conf_s2_clamped) * torch.log(1.0 - conf_s2_clamped))
            loss_conf_half_entropy = ((0.6931 - conf_s2_entropy) * inv_occ_s2).sum() / (inv_occ_s2.sum() + 1e-6)
        fused_s2, _ = self.fusion_1_2(
            curr_s2,
            fused_s3_up,
            conf_s2_fuse,
            inv_mask=inv_s2,
            occ_mask=occ_s2,
            precomputed=(fused_aff_s2, refined_hist_s2),
        )

        curr_s1 = mlvl_curr_feats[1]
        fused_s2_up = F.interpolate(fused_s2, size=curr_s1.shape[-3:], mode="trilinear", align_corners=False)
        fused_aff_s1, refined_hist_s1 = self.fusion_1_1.compute_affinity(curr_s1, fused_s2_up)
        conf_s1 = self.conf_full_head(torch.cat([curr_s1, fused_aff_s1], dim=1))
        inv_s1 = self._resize_mask(inv_s3, curr_s1.shape[-3:])
        occ_s1 = self._resize_mask(occ_s3, curr_s1.shape[-3:])
        inv_occ_s1 = (inv_s1 | occ_s1).unsqueeze(0).unsqueeze(0).to(conf_s1.dtype)
        conf_s1_fuse = conf_s1 * inv_occ_s1
        conf_s1_target = inv_occ_s1
        if gt_occ is not None:
            conf_s1_loss_map = F.binary_cross_entropy(conf_s1, conf_s1_target, reduction="none")
            loss_conf_full = (conf_s1_loss_map * inv_occ_s1).sum() / (inv_occ_s1.sum() + 1e-6)
            conf_s1_clamped = conf_s1.clamp(min=eps, max=1.0 - eps)
            conf_s1_entropy = -(conf_s1_clamped * torch.log(conf_s1_clamped) + (1.0 - conf_s1_clamped) * torch.log(1.0 - conf_s1_clamped))
            loss_conf_full_entropy = ((0.6931 - conf_s1_entropy) * inv_occ_s1).sum() / (inv_occ_s1.sum() + 1e-6)
        fused_s1, _ = self.fusion_1_1(
            curr_s1,
            fused_s2_up,
            conf_s1_fuse,
            inv_mask=inv_s1,
            occ_mask=occ_s1,
            precomputed=(fused_aff_s1, refined_hist_s1),
        )

        curr_s0 = mlvl_curr_feats[0]
        fused_s1_up = F.interpolate(fused_s1, size=curr_s0.shape[-3:], mode="trilinear", align_corners=False)
        fused_aff_s0, refined_hist_s0 = self.fusion_1_0.compute_affinity(curr_s0, fused_s1_up)
        conf_s0 = self.conf_final_head(torch.cat([curr_s0, fused_aff_s0], dim=1))
        inv_s0 = self._resize_mask(inv_s3, curr_s0.shape[-3:])
        occ_s0 = self._resize_mask(occ_s3, curr_s0.shape[-3:])
        inv_occ_s0 = (inv_s0 | occ_s0).unsqueeze(0).unsqueeze(0).to(conf_s0.dtype)
        conf_s0_fuse = conf_s0 * inv_occ_s0
        conf_s0_target = inv_occ_s0
        if gt_occ is not None:
            conf_s0_loss_map = F.binary_cross_entropy(conf_s0, conf_s0_target, reduction="none")
            loss_conf_final = (conf_s0_loss_map * inv_occ_s0).sum() / (inv_occ_s0.sum() + 1e-6)
            conf_s0_clamped = conf_s0.clamp(min=eps, max=1.0 - eps)
            conf_s0_entropy = -(conf_s0_clamped * torch.log(conf_s0_clamped) + (1.0 - conf_s0_clamped) * torch.log(1.0 - conf_s0_clamped))
            loss_conf_final_entropy = ((0.6931 - conf_s0_entropy) * inv_occ_s0).sum() / (inv_occ_s0.sum() + 1e-6)
        fused_s0, _ = self.fusion_1_0(
            curr_s0,
            fused_s1_up,
            conf_s0_fuse,
            inv_mask=inv_s0,
            occ_mask=occ_s0,
            precomputed=(fused_aff_s0, refined_hist_s0),
        )

        fused_feats = [fused_s0, fused_s1, fused_s2, fused_s3]
        flow_loss_dict = {}
        if gt_occ is not None:
            aux_loss_weight = self._get_aux_loss_weight()
            runner = getattr(self, "runner", None)
            current_epoch = getattr(runner, "epoch", None)
            if self.training and current_epoch is not None:
                if getattr(self, "_aux_w_logged_epoch", None) != current_epoch:
                    msg = f"[aux] epoch={current_epoch + 1}, aux_w={aux_loss_weight:.6f}"
                    if hasattr(self, "logger") and self.logger is not None:
                        self.logger.info(msg)
                    else:
                        print(msg)
                    self._aux_w_logged_epoch = current_epoch
            for k, v in aux_losses.items():
                flow_loss_dict[k] = v * aux_loss_weight

            flow_loss_dict["loss_align"] = loss_align * self.loss_align_weight
            flow_loss_dict["loss_conf"] = loss_conf * self.loss_conf_weight
            flow_loss_dict["loss_conf_entropy"] = loss_conf_entropy * self.loss_conf_entropy_weight
            flow_loss_dict["loss_conf_half"] = loss_conf_half * self.loss_conf_half_weight
            flow_loss_dict["loss_conf_full"] = loss_conf_full * self.loss_conf_full_weight
            flow_loss_dict["loss_conf_half_entropy"] = loss_conf_half_entropy * self.loss_conf_half_entropy_weight
            flow_loss_dict["loss_conf_full_entropy"] = loss_conf_full_entropy * self.loss_conf_full_entropy_weight
            flow_loss_dict["loss_conf_final"] = loss_conf_final * self.loss_conf_final_weight
            flow_loss_dict["loss_conf_final_entropy"] = loss_conf_final_entropy * self.loss_conf_final_entropy_weight
            if loss_flow_rigid is not None:
                flow_loss_dict["loss_flow_rigid"] = loss_flow_rigid * self.loss_flow_rigid_weight
        return fused_feats, flow_loss_dict

    @force_fp32(apply_to=("pts_feats",))
    def forward_pts_train(
        self,
        pts_feats,
        gt_occ=None,
        points_occ=None,
        img_metas=None,
        img_feats=None,
        points_uv=None,
        **kwargs,
    ):
        losses, query_scores, query_feats = self.pts_bbox_head.forward_train(
            voxel_feats=pts_feats,
            img_metas=img_metas,
            gt_occ=gt_occ,
            points=points_occ,
            img_feats=img_feats,
            points_uv=points_uv,
            sem_dic=self.sem_dic.clone(),
            sem_embed=self.dic_embed.weight,
            **kwargs,
        )
        return losses, query_scores, query_feats

    def forward_train(
        self,
        points=None,
        img_metas=None,
        img_inputs=None,
        gt_occ=None,
        points_occ=None,
        points_uv=None,
        **kwargs,
    ):
        (
            mlvl_curr_feats,
            mlvl_prev_feats,
            _,
            mlvl_inv_masks,
            mlvl_occ_masks,
            curr_geo_inputs,
        ) = self.extract_img_feat(img_inputs, img_metas)

        fused_feats, flow_losses = self._flow_fuse_multiscale(
            mlvl_curr_feats,
            mlvl_prev_feats,
            mlvl_inv_masks,
            mlvl_occ_masks,
            gt_occ=gt_occ,
            geo_inputs=curr_geo_inputs,
            img_metas=img_metas,
        )
        fused_masks = self._build_masks_from_feats(fused_feats)
        voxel_feats = self.bev_encoder(fused_feats, curr_geo_inputs, fused_masks)

        losses = {}
        losses_occupancy, query_scores, query_feats = self.forward_pts_train(
            voxel_feats,
            gt_occ=gt_occ,
            points_occ=points_occ,
            img_metas=img_metas,
            points_uv=points_uv,
            **kwargs,
        )
        losses.update(losses_occupancy)
        flow_aux_weight = self._get_flow_aux_weight()
        runner = getattr(self, "runner", None)
        current_epoch = getattr(runner, "epoch", None)
        if self.training and current_epoch is not None:
            if getattr(self, "_flow_aux_w_logged_epoch", None) != current_epoch:
                msg = f"[flow_aux] epoch={current_epoch + 1}, flow_aux_w={flow_aux_weight:.6f}"
                if hasattr(self, "logger") and self.logger is not None:
                    self.logger.info(msg)
                else:
                    print(msg)
                self._flow_aux_w_logged_epoch = current_epoch

        # Monitoring only: compare flow-aux magnitude against main Mask2Former losses.
        main_total = None
        for v in losses_occupancy.values():
            if torch.is_tensor(v):
                main_total = v if main_total is None else (main_total + v)
        flow_total_raw = None
        for v in flow_losses.values():
            if torch.is_tensor(v):
                flow_total_raw = v if flow_total_raw is None else (flow_total_raw + v)
        if main_total is not None and flow_total_raw is not None:
            denom = main_total.detach().abs() + 1e-6
            losses["flow_aux_ratio_raw"] = flow_total_raw.detach() / denom
            losses["flow_aux_ratio_scaled"] = (flow_total_raw.detach() * flow_aux_weight) / denom

        for k, v in flow_losses.items():
            losses[k] = v * flow_aux_weight

        query_scores = F.softmax(query_scores, dim=-1)[..., :-1]
        self.update_dict(query_feats, query_scores)
        return losses

    def simple_test(self, img_metas, img=None, rescale=False, points_occ=None, gt_occ=None, points_uv=None):
        (
            mlvl_curr_feats,
            mlvl_prev_feats,
            _,
            mlvl_inv_masks,
            mlvl_occ_masks,
            curr_geo_inputs,
        ) = self.extract_img_feat(img, img_metas)
        fused_feats, _ = self._flow_fuse_multiscale(
            mlvl_curr_feats,
            mlvl_prev_feats,
            mlvl_inv_masks,
            mlvl_occ_masks,
            gt_occ=None,
            geo_inputs=curr_geo_inputs,
            img_metas=img_metas,
        )
        fused_masks = self._build_masks_from_feats(fused_feats)
        voxel_feats = self.bev_encoder(fused_feats, curr_geo_inputs, fused_masks)

        output, query_scores, query_feats = self.pts_bbox_head.simple_test(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=None,
            points_uv=points_uv,
            sem_dic=self.sem_dic.clone(),
            sem_embed=self.dic_embed.weight,
        )

        output_voxels = output["output_voxels"][0]
        target_occ_size = img_metas[0]["occ_size"]
        if tuple(output_voxels.shape[-3:]) != tuple(target_occ_size):
            output_voxels = F.interpolate(
                output_voxels, size=tuple(target_occ_size), mode="trilinear", align_corners=True
            )
        output["output_voxels"] = output_voxels
        output["target_voxels"] = gt_occ

        if self.update_for_infer:
            query_scores = F.softmax(query_scores, dim=-1)[..., :-1]
            self.update_dict(query_feats, query_scores)
        return output

    def update_dict(self, query, scores):
        scores = scores.clone().detach().flatten(0, 1).T
        query = query.clone().detach().flatten(0, 1)

        repo_feats = self.sem_dic.clone().mean(1)
        query_norm = F.normalize(query, dim=1)
        repo_norm = F.normalize(repo_feats, dim=1)
        novelty = -torch.mm(repo_norm, query_norm.T)
        query_scores = scores + 0.2 * novelty
        metrics = torch.cat([query_scores, self.sem_dic_scores], dim=1)
        new_scores, topk_ind = torch.topk(metrics, k=10, dim=1)
        query = query.unsqueeze(0).expand(self.sem_dic.shape[0], -1, -1)
        temp_query = torch.cat([query, self.sem_dic], dim=1)
        topk_ind2 = topk_ind.unsqueeze(-1).expand(-1, -1, query.shape[-1])
        updated_dict = torch.gather(temp_query, dim=1, index=topk_ind2)

        self.sem_dic.copy_(updated_dict.detach())
        self.sem_dic_scores.copy_(new_scores.detach())

    def _resize_gt(self, gt_occ, size):
        gt = gt_occ.unsqueeze(1).float()
        gt = F.interpolate(gt, size=size, mode="nearest")
        return gt.squeeze(1).long()

    def _resize_mask(self, mask, size):
        mask = mask.float().unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=size, mode="nearest")
        mask = mask.squeeze(0).squeeze(0).bool()
        return mask

    def _build_masks_from_feats(self, mlvl_feats):
        # Recompute empty-voxel masks after flow fusion for downstream BEV encoder.
        masks = []
        with torch.no_grad():
            for feat in mlvl_feats:
                voxel_mask = (feat.detach().sum(1) == 0).to(torch.uint8)
                masks.append(voxel_mask)
        return masks

    def _loss_ssc(self, pred, target, prefix=""):
        loss_dict = {}
        class_weights = None
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device=pred.device, dtype=pred.dtype)
        loss_dict[f"loss_{prefix}ce"] = self.loss_weight_cfg["loss_voxel_ce_weight"] * CE_ssc_loss(
            pred, target, class_weights, ignore_index=255
        )
        loss_dict[f"loss_{prefix}sem_scal"] = self.loss_weight_cfg["loss_voxel_sem_scal_weight"] * sem_scal_loss(
            pred, target, ignore_index=255
        )
        loss_dict[f"loss_{prefix}geo_scal"] = self.loss_weight_cfg["loss_voxel_geo_scal_weight"] * geo_scal_loss(
            pred, target, ignore_index=255, non_empty_idx=self.empty_idx
        )
        return loss_dict

