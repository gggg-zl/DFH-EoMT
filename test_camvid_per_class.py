# ---------------------------------------------------------------
# Standalone script: evaluate a trained CamVid model on the test
# set and print per-class IoU for all 11 classes.
#
# Usage (same style as main.py):
#   python test_camvid_per_class.py \
#       -c configs/dinov2/camvid/semantic/eomt_large_720.yaml \
#       --model.ckpt_path checkpoints/best_model.ckpt \
#       --data.path       camvid/
# ---------------------------------------------------------------

import sys
import os

# ── pre-process dot-notation args before argparse sees them ───────────────────
# Converts  --data.path X  →  --data_path X
#           --model.ckpt_path X  →  --ckpt X
_argv = []
_skip = False
for _i, _tok in enumerate(sys.argv[1:], 1):
    if _skip:
        _argv.append(_tok)
        _skip = False
        continue
    if _tok == "--data.path":
        _argv.append("--data_path")
        _skip = True
    elif _tok == "--model.ckpt_path":
        _argv.append("--ckpt")
        _skip = True
    elif _tok.startswith("--data.path="):
        _argv.append("--data_path=" + _tok.split("=", 1)[1])
    elif _tok.startswith("--model.ckpt_path="):
        _argv.append("--ckpt=" + _tok.split("=", 1)[1])
    else:
        _argv.append(_tok)
sys.argv[1:] = _argv

import argparse
import torch
import torch.nn.functional as F

os.environ["TORCH_LOGS"] = "-dynamo"
torch._dynamo.config.suppress_errors = True  # type: ignore[attr-defined]

from torchmetrics.classification import MulticlassJaccardIndex

from datasets.camvid_semantic import CamVidSemantic
from training.mask_classification_semantic import MaskClassificationSemantic

CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Pavement",
    "Tree", "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
]
NUM_CLASSES = 11
IGNORE_IDX  = 255

bold_green = "\033[1;32m"
reset_col   = "\033[0m"


# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, config_path: str) -> MaskClassificationSemantic:
    """Load a MaskClassificationSemantic model from a Lightning checkpoint.

    The YAML config is only used to build the network architecture; weights
    are then overwritten by the checkpoint.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    # Resolve data-linked arguments that main.py wires automatically
    data_cfg  = cfg.get("data", {}).get("init_args", {})
    num_classes = data_cfg.get("num_classes", NUM_CLASSES)
    img_size    = data_cfg.get("img_size", [360, 480])

    model_init = model_cfg["init_args"]
    model_init["num_classes"] = num_classes

    # propagate into network as main.py's link_arguments does
    # (note: EoMT does NOT take img_size, only the encoder does)
    net_init = model_init["network"]["init_args"]
    net_init["num_classes"] = num_classes

    # propagate into encoder – create init_args if it doesn't exist in YAML
    encoder_spec = net_init.get("encoder")
    if isinstance(encoder_spec, dict):
        if "init_args" not in encoder_spec:
            encoder_spec["init_args"] = {}
        encoder_spec["init_args"]["img_size"]   = img_size
        # pass ckpt_path=None so ViT won't try to load DINOv2 weights;
        # the full fine-tuned weights come from the Lightning checkpoint below
        encoder_spec["init_args"].setdefault("ckpt_path", None)

    # Build network
    import importlib

    def _instantiate(spec):
        module_path, cls_name = spec["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        raw_args = spec.get("init_args", {})
        # recursively instantiate nested class specs
        resolved = {}
        for k, v in raw_args.items():
            if isinstance(v, dict) and "class_path" in v:
                resolved[k] = _instantiate(v)
            else:
                resolved[k] = v
        return cls(**resolved)

    network = _instantiate(model_cfg["init_args"]["network"])

    model = MaskClassificationSemantic(
        network=network,
        img_size=tuple(img_size),
        num_classes=num_classes,
        attn_mask_annealing_enabled=model_init.get("attn_mask_annealing_enabled", False),
        attn_mask_annealing_start_steps=model_init.get("attn_mask_annealing_start_steps"),
        attn_mask_annealing_end_steps=model_init.get("attn_mask_annealing_end_steps"),
        ckpt_path=ckpt_path,
    )
    model.eval()
    return model


def build_dataloader(config_path: str, data_path: str | None):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_init = cfg.get("data", {}).get("init_args", {})
    if data_path:
        data_init["path"] = data_path

    dm = CamVidSemantic(**data_init)
    dm.setup()
    return dm.test_dataloader()


def to_per_pixel_targets(targets, ignore_idx=IGNORE_IDX):
    per_pixel = []
    for t in targets:
        ppt = torch.full(
            t["masks"].shape[-2:], ignore_idx,
            dtype=t["labels"].dtype, device=t["labels"].device,
        )
        for i, mask in enumerate(t["masks"]):
            ppt[mask] = t["labels"][i]
        per_pixel.append(ppt)
    return per_pixel


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CamVid per-class IoU on test set")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    parser.add_argument("--ckpt",         required=True, help="Path to .ckpt checkpoint  (or --model.ckpt_path)")
    parser.add_argument("--data_path",    default=None,  help="Override dataset root path (or --data.path)")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── load model ─────────────────────────────────────────────────────────────
    print("Loading model …")
    model = load_model(args.ckpt, args.config)
    model = model.to(device)
    model.eval()

    # ── load test data ─────────────────────────────────────────────────────────
    print("Building test dataloader …")
    loader = build_dataloader(args.config, args.data_path)
    print(f"Test set size: {len(loader.dataset)} images")

    # ── metric ─────────────────────────────────────────────────────────────────
    metric = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES,
        validate_args=False,
        ignore_index=IGNORE_IDX,
        average=None,           # returns IoU per class
    ).to(device)

    # ── inference loop ─────────────────────────────────────────────────────────
    print("Running inference …")
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader):
            imgs    = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            img_sizes = [img.shape[-2:] for img in imgs]
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_all, class_logits_all = model(crops)

            gt_pixels = to_per_pixel_targets(targets)

            # use only the last (or only) decoder block
            mask_logits  = mask_logits_all[-1]
            class_logits = class_logits_all[-1]

            mask_logits  = F.interpolate(mask_logits, model.img_size, mode="bilinear")
            crop_logits  = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits_list  = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            for pred_logits, gt in zip(logits_list, gt_pixels):
                metric.update(pred_logits[None], gt[None])

            if (batch_idx + 1) % 20 == 0:
                print(f"  processed {batch_idx + 1}/{len(loader)} batches")

    # ── compute & display ──────────────────────────────────────────────────────
    iou_per_class = metric.compute()          # shape: (11,)
    miou = float(iou_per_class.mean())

    sep = "-" * 28
    print(f"\n{bold_green}{'Class':<16} {'IoU (%)':>8}{reset_col}")
    print(sep)
    for i, name in enumerate(CLASS_NAMES):
        iou_val = float(iou_per_class[i]) * 100
        print(f"{name:<16} {iou_val:>7.1f}%")
    print(sep)
    print(f"{bold_green}{'mIoU':<16} {miou * 100:>7.1f}%{reset_col}\n")


if __name__ == "__main__":
    main()
