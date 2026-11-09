#!/usr/bin/env python3
import argparse
import importlib
import os
import sys
from typing import Any

torch = None
Config = None
build_dataloader = None
build_dataset = None
build_model = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke check for FFPS-enhanced model (forward/backward + loss sanity)."
    )
    parser.add_argument(
        "--config",
        default="projects/configs/ffps_kitti.py",
        help="Config file path.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="How many mini-batches to run.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers per GPU.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data_root in cfg.data.{train,val,test}.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split used for smoke check.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Tiny LR for optional optimizer step.",
    )
    parser.add_argument(
        "--no-step",
        action="store_true",
        help="Only backward, do not optimizer.step().",
    )
    parser.add_argument(
        "--no-strict-keys",
        action="store_true",
        help="Do not fail when required loss keys are missing.",
    )
    parser.add_argument(
        "--no-freeze-bn",
        action="store_true",
        help="Keep BatchNorm layers in train mode (default is freezing BN to eval for batch-size-1 stability).",
    )
    return parser.parse_args()


def ensure_repo_on_path(repo_root: str):
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    soap_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "soap")
    if soap_dir not in sys.path:
        sys.path.insert(0, soap_dir)
    dcn_dir = os.path.join(soap_dir, "dcn")
    if dcn_dir not in sys.path:
        sys.path.insert(0, dcn_dir)


def import_plugins(cfg: Any, config_path: str):
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        plugin_dir = cfg.plugin_dir if hasattr(cfg, "plugin_dir") else os.path.dirname(config_path)
        module_path = plugin_dir.rstrip("/\\").replace("/", ".").replace("\\", ".")
        importlib.import_module(module_path)


def maybe_override_data_root(cfg: Config, data_root: str):
    if not data_root or not hasattr(cfg, "data"):
        return
    for split in ["train", "val", "test"]:
        if split in cfg.data:
            cfg.data[split].data_root = data_root
            if hasattr(cfg.data[split], "ann_file"):
                cfg.data[split].ann_file = os.path.join(data_root, "dataset", "labels")
            if hasattr(cfg.data[split], "pipeline"):
                for step in cfg.data[split].pipeline:
                    if isinstance(step, dict) and "stereo_path" in step:
                        step["stereo_path"] = os.path.join(data_root, "dataset", "stereodepth")


def unwrap_data_container(x: Any):
    return x.data[0] if hasattr(x, "data") else x


def to_device(x: Any, device: str):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def ensure_batch_dim(x: Any):
    if torch.is_tensor(x):
        if x.dim() in (2, 3, 4) and x.size(0) != 1:
            return x.unsqueeze(0)
        return x
    if isinstance(x, (list, tuple)):
        return type(x)(ensure_batch_dim(v) for v in x)
    if isinstance(x, dict):
        return {k: ensure_batch_dim(v) for k, v in x.items()}
    return x


def check_required_keys(losses: dict, strict: bool):
    required = [
        "loss_cls",
        "loss_mask",
        "loss_dice",
        "loss_aux_ce",
        "loss_aux_sem_scal",
        "loss_aux_geo_scal",
        "loss_align",
        "loss_conf",
        "loss_conf_half",
        "loss_conf_full",
        "loss_conf_final",
    ]
    missing = [k for k in required if k not in losses]
    if missing:
        msg = f"Missing required loss keys: {missing}"
        if strict:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")


def check_finite_losses(losses: dict):
    bad = []
    for k, v in losses.items():
        if torch.is_tensor(v):
            if not torch.isfinite(v).all():
                bad.append(k)
    if bad:
        raise RuntimeError(f"Found NaN/Inf in loss keys: {bad}")


def parse_total_loss(model, losses: dict):
    if hasattr(model, "_parse_losses"):
        total_loss, _ = model._parse_losses(losses)
        return total_loss

    total = None
    for k, v in losses.items():
        if "loss" in k and torch.is_tensor(v):
            total = v if total is None else (total + v)
    if total is None:
        raise RuntimeError("Could not aggregate total loss from loss dict.")
    return total


def freeze_batchnorm_layers(model):
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def main():
    global torch, Config, build_dataloader, build_dataset, build_model
    args = parse_args()

    try:
        import torch as _torch
        from mmcv import Config as _Config
        from mmdet3d.datasets import build_dataloader as _build_dataloader
        from mmdet3d.datasets import build_dataset as _build_dataset
        from mmdet3d.models import build_model as _build_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing runtime dependency. Please run in the training environment with torch/mmcv/mmdet3d installed."
        ) from exc

    torch = _torch
    Config = _Config
    build_dataloader = _build_dataloader
    build_dataset = _build_dataset
    build_model = _build_model

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ensure_repo_on_path(repo_root)

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(repo_root, cfg_path)
    cfg = Config.fromfile(cfg_path)
    import_plugins(cfg, cfg_path)
    maybe_override_data_root(cfg, args.data_root)

    dataset_cfg = cfg.data.train if args.split == "train" else cfg.data.val
    dataset = build_dataset(dataset_cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.num_workers,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.to(args.device)
    model.train()
    if not args.no_freeze_bn:
        bn_count = freeze_batchnorm_layers(model)
        print(f"[smoke] BatchNorm layers set to eval: {bn_count}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    strict_keys = not args.no_strict_keys
    data_iter = iter(data_loader)
    for i in range(args.iters):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)

        img_metas = unwrap_data_container(data.get("img_metas")) if "img_metas" in data else None
        img_inputs = unwrap_data_container(data.get("img_inputs")) if "img_inputs" in data else None
        gt_occ = unwrap_data_container(data.get("gt_occ")) if "gt_occ" in data else None

        if isinstance(img_inputs, dict) and "img_inputs" in img_inputs:
            img_inputs = img_inputs["img_inputs"]
        if isinstance(img_inputs, list) and len(img_inputs) == 1 and isinstance(img_inputs[0], (list, tuple)):
            img_inputs = img_inputs[0]
        if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], list):
            img_metas = img_metas[0]
        if img_metas is not None and not isinstance(img_metas, list):
            img_metas = [img_metas]

        img_inputs = ensure_batch_dim(to_device(img_inputs, args.device))
        if gt_occ is not None and torch.is_tensor(gt_occ):
            gt_occ = gt_occ.to(args.device)

        optimizer.zero_grad(set_to_none=True)
        losses = model.forward_train(
            img_metas=img_metas,
            img_inputs=img_inputs,
            gt_occ=gt_occ,
        )

        if not isinstance(losses, dict):
            raise RuntimeError(f"forward_train should return dict, got {type(losses)}")
        check_required_keys(losses, strict=strict_keys)
        check_finite_losses(losses)

        total_loss = parse_total_loss(model, losses)
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Total loss is invalid: {total_loss}")
        total_loss.backward()
        if not args.no_step:
            optimizer.step()

        msg = f"[iter {i + 1}/{args.iters}] total_loss={float(total_loss.detach().cpu().item()):.6f}"
        if "flow_aux_ratio_raw" in losses and torch.is_tensor(losses["flow_aux_ratio_raw"]):
            msg += f", flow_aux_ratio_raw={float(losses['flow_aux_ratio_raw'].detach().cpu().item()):.6f}"
        if "flow_aux_ratio_scaled" in losses and torch.is_tensor(losses["flow_aux_ratio_scaled"]):
            msg += f", flow_aux_ratio_scaled={float(losses['flow_aux_ratio_scaled'].detach().cpu().item()):.6f}"
        print(msg)

    print("Smoke check passed.")


if __name__ == "__main__":
    main()

