#!/usr/bin/env python3
import argparse
import os
import sys

import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal forward shape check for FFPS.")
    parser.add_argument(
        "--config",
        default="projects/configs/ffps_kitti.py",
        help="Config file path.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="test",
        help="Use train or test pipeline.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data_root in config (e.g. MyModel/data/SemanticKITTI).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Ensure project root is on sys.path so "projects.*" imports work.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    # Ensure dcn package (under projects/mmdet3d_plugin/soap) is discoverable.
    soap_dir = os.path.join(repo_root, "projects", "mmdet3d_plugin", "soap")
    if soap_dir not in sys.path:
        sys.path.insert(0, soap_dir)
    dcn_dir = os.path.join(soap_dir, "dcn")
    if dcn_dir not in sys.path:
        sys.path.insert(0, dcn_dir)
    cfg = Config.fromfile(args.config)
    # Ensure custom datasets/modules are registered (avoid importing full soap package).
    try:
        import projects.mmdet3d_plugin.datasets  # noqa: F401
        import projects.mmdet3d_plugin.soap.detectors.ffps  # noqa: F401
        import projects.mmdet3d_plugin.soap.image2bev.ViewTransformerLSSVoxel  # noqa: F401
        import projects.mmdet3d_plugin.soap.flow.resnet3d  # noqa: F401
        import projects.mmdet3d_plugin.soap.flow.pred_block  # noqa: F401
        import projects.mmdet3d_plugin.soap.flow.fpn3d  # noqa: F401
        import projects.mmdet3d_plugin.soap.flow.flow_head  # noqa: F401
    except Exception as exc:
        print("Warning: failed to import custom project modules:", exc)

    if args.data_root:
        if hasattr(cfg, "data"):
            for split in ["train", "val", "test"]:
                if split in cfg.data:
                    cfg.data[split].data_root = args.data_root
                    # Update ann_file if it is relative to old root.
                    if hasattr(cfg.data[split], "ann_file"):
                        cfg.data[split].ann_file = os.path.join(args.data_root, "dataset", "labels")
                    # Update stereo depth path inside pipelines if present.
                    if hasattr(cfg.data[split], "pipeline"):
                        for step in cfg.data[split].pipeline:
                            if isinstance(step, dict) and "stereo_path" in step:
                                step["stereo_path"] = os.path.join(args.data_root, "dataset", "stereodepth")

    if args.mode == "train":
        dataset_cfg = cfg.data.train
    else:
        dataset_cfg = cfg.data.val

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
    model.eval()

    data = next(iter(data_loader))
    # Move data to device (mmdet handles data containers when calling model)
    # unwrap DataContainer for single-sample forward
    def _unwrap(dc):
        return dc.data[0] if hasattr(dc, "data") else dc

    img_metas = _unwrap(data.get("img_metas")) if "img_metas" in data else None
    img_inputs = _unwrap(data.get("img_inputs")) if "img_inputs" in data else None
    gt_occ = _unwrap(data.get("gt_occ")) if "gt_occ" in data else None

    # normalize img_inputs structure (should be tuple/list of tensors)
    if isinstance(img_inputs, dict) and "img_inputs" in img_inputs:
        img_inputs = img_inputs["img_inputs"]
    if isinstance(img_inputs, list) and len(img_inputs) == 1 and isinstance(img_inputs[0], (list, tuple)):
        img_inputs = img_inputs[0]

    # normalize img_metas to list
    if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], list):
        img_metas = img_metas[0]
    if img_metas is not None and not isinstance(img_metas, list):
        img_metas = [img_metas]

    # move tensors in img_inputs to device
    def _to_device(x, device):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, (list, tuple)):
            return type(x)(_to_device(v, device) for v in x)
        return x

    if img_inputs is not None:
        # ensure batch dim exists on each tensor in img_inputs
        def _ensure_batch(t):
            if not torch.is_tensor(t):
                return t
            # Heuristics: add batch dim if missing
            if t.dim() in (2, 3, 4) and t.size(0) != 1:
                return t.unsqueeze(0)
            return t

        if isinstance(img_inputs, (list, tuple)):
            img_inputs = type(img_inputs)(_ensure_batch(_to_device(v, args.device)) for v in img_inputs)
        else:
            img_inputs = _ensure_batch(_to_device(img_inputs, args.device))
    if gt_occ is not None and torch.is_tensor(gt_occ):
        gt_occ = gt_occ.to(args.device)

    if args.mode == "train":
        # Call forward_train directly with prepared tensors (keep grads to allow debug prints).
        outputs = model.forward_train(
            img_metas=img_metas,
            img_inputs=img_inputs,
            gt_occ=gt_occ,
        )
        print("Train forward ok. Loss keys:", list(outputs.keys()))
        if isinstance(outputs, dict) and "loss_align" in outputs:
            try:
                print("loss_align:", float(outputs["loss_align"].detach().cpu().item()))
            except Exception:
                pass
    else:
        with torch.no_grad():
            # Call simple_test directly to avoid augmentation packaging issues.
            outputs = model.simple_test(img_metas, img_inputs, gt_occ=gt_occ)
            print("Test forward ok.")

    if isinstance(outputs, dict) and "output_voxels" in outputs:
        out = outputs["output_voxels"]
        if torch.is_tensor(out):
            print("output_voxels shape:", tuple(out.shape))
        elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
            print("output_voxels[0] shape:", tuple(out[0].shape))
        else:
            print("output_voxels type:", type(out))

    if isinstance(outputs, dict) and "target_voxels" in outputs and torch.is_tensor(outputs["target_voxels"]):
        print("target_voxels shape:", tuple(outputs["target_voxels"].shape))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()


