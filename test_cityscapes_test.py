# ---------------------------------------------------------------
# Batch inference script for Cityscapes test set.
# Saves single-channel labelId PNGs matching the official submission format.
#
# Official requirements (from cityscapesscripts evaluation script):
#   - Pixel values = Cityscapes labelId (0-33), NOT train_id (0-18)
#   - Single-channel PNG, 2048x1024 resolution
#   - Use original leftImg8bit filenames (e.g. berlin_000000_000019_leftImg8bit.png)
#   - Flat zip archive, no subdirectories
#
# Usage:
#   python test_cityscapes_test.py \
#       -c configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml \
#       --ckpt checkpoints/best_model-cityscapes.ckpt \
#       --data_path city/ \
#       --output_dir cityscapes_test_pred
# ---------------------------------------------------------------

import sys
import os

# Pre-process dot-notation args before argparse sees them.
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

from pathlib import Path

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, desc=""):
        return x

os.environ["TORCH_LOGS"] = "-dynamo"
torch._dynamo.config.suppress_errors = True

from training.mask_classification_semantic import MaskClassificationSemantic


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, config_path: str) -> MaskClassificationSemantic:
    import yaml
    import importlib

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg  = cfg.get("data", {}).get("init_args", {})
    num_classes = data_cfg.get("num_classes", 19)
    img_size    = data_cfg.get("img_size", [1024, 512])

    model_init = model_cfg["init_args"]
    model_init["num_classes"] = num_classes

    net_init = model_init["network"]["init_args"]
    net_init["num_classes"] = num_classes

    encoder_spec = net_init.get("encoder")
    if isinstance(encoder_spec, dict):
        if "init_args" not in encoder_spec:
            encoder_spec["init_args"] = {}
        encoder_spec["init_args"]["img_size"] = list(img_size)
        encoder_spec["init_args"].setdefault("ckpt_path", None)

    def _instantiate(spec):
        module_path, cls_name = spec["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        raw_args = spec.get("init_args", {})
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


def build_test_loader(config_path: str, data_path: str | None):
    import yaml
    from torch.utils.data import DataLoader
    from torchvision.datasets import Cityscapes

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_init = cfg.get("data", {}).get("init_args", {})
    if data_path:
        data_init["path"] = data_path

    img_size = data_init.get("img_size", [1024, 512])
    if isinstance(img_size, list):
        img_size = tuple(img_size)

    # torchvision.datasets.Cityscapes applies transforms internally in __getitem__.
    # We subclass to bypass the mask target and apply our own resize.
    class CityscapesImageOnly(Cityscapes):
        def __init__(self, *args, img_size, **kwargs):
            super().__init__(*args, **kwargs)
            from torchvision.transforms import v2 as T
            h, w = img_size[1], img_size[0]
            self.img_resize = T.Resize((h, w), antialias=True)

        def __getitem__(self, index):
            import torchvision.transforms.functional as TF
            img, _ = super().__getitem__(index)  # img is already PIL here
            orig_h, orig_w = img.height, img.width
            img = self.img_resize(img)
            img_tensor = TF.to_tensor(img)
            img_uint8 = (img_tensor * 255).to(torch.uint8)
            return img_uint8, (orig_h, orig_w)

    cityscapes_test = CityscapesImageOnly(
        root=data_init["path"],
        split="test",
        target_type=["polygon", "semantic"],
        img_size=img_size,
    )

    def eval_collate(batch):
        imgs, sizes = zip(*batch)
        return imgs, sizes

    num_workers = data_init.get("num_workers", 4)
    return DataLoader(
        cityscapes_test,
        batch_size=1,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=num_workers,
        pin_memory=True,
    )


def run_inference(model, loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # torchvision.datasets.Cityscapes stores image paths in .images attribute
    test_dataset = loader.dataset
    image_paths = test_dataset.images  # list of absolute file paths

    num_samples = len(test_dataset)
    sample_idx = 0

    # Cityscapes official submission: single-channel PNG, labelId (0-33), original filenames.
    # See cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py:
    #   "Make sure that our results use the original IDs and not the training IDs."
    TRAIN_ID_TO_LABEL_ID = {
        0: 7,   # road
        1: 8,   # sidewalk
        2: 11,  # building
        3: 12,  # wall
        4: 13,  # fence
        5: 17,  # pole
        6: 19,  # traffic light
        7: 20,  # traffic sign
        8: 21,  # vegetation
        9: 22,  # terrain
        10: 23, # sky
        11: 24, # person
        12: 25, # rider
        13: 26, # car
        14: 27, # truck
        15: 28, # bus
        16: 31, # train
        17: 32, # motorcycle
        18: 33, # bicycle
    }

    with torch.no_grad():
        for batch_idx, (imgs_tuple, sizes_tuple) in enumerate(tqdm(loader, desc="Inference")):
            for img_tensor, (orig_h, orig_w) in zip(imgs_tuple, sizes_tuple):
                img_device = img_tensor.unsqueeze(0).to(device)

                crops, origins = model.window_imgs_semantic([img_device[0]])
                mask_logits_all, class_logits_all = model(crops)

                mask_logits  = mask_logits_all[-1]
                class_logits = class_logits_all[-1]

                mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")
                crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)

                logits_list = model.revert_window_logits_semantic(
                    crop_logits, origins, [(orig_h, orig_w)]
                )
                pred_logits = logits_list[0]

                pred_resized = F.interpolate(
                    pred_logits.unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                )[0]
                pred_train_id = pred_resized.argmax(dim=0).cpu().numpy().astype(np.uint8)

                # Map train_id (0-18) back to Cityscapes labelId (0-33)
                pred_labelId = np.full_like(pred_train_id, 255)
                for train_id, label_id in TRAIN_ID_TO_LABEL_ID.items():
                    pred_labelId[pred_train_id == train_id] = label_id

                # Use the original image filename (e.g. berlin_000000_000019_leftImg8bit.png).
                # The eval server searches recursively and matches by city_seq_frame prefix.
                orig_path = Path(image_paths[sample_idx])
                filename = orig_path.name  # full original filename with .png
                save_path = os.path.join(output_dir, filename)
                Image.fromarray(pred_labelId).save(save_path)

                sample_idx += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {sample_idx} / {num_samples} images")

    assert sample_idx == num_samples, f"Expected {num_samples} samples, got {sample_idx}"


def main():
    parser = argparse.ArgumentParser(description="Cityscapes test set inference")
    parser.add_argument("-c", "--config",    required=True, help="Path to YAML config")
    parser.add_argument("--ckpt",            required=True, help="Path to .ckpt checkpoint")
    parser.add_argument("--data_path",       default=None,  help="Dataset root (contains zip files)")
    parser.add_argument("--output_dir",      default="./cityscapes_test_pred",
                        help="Output directory for predictions")
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("Loading model ...")
    model = load_model(args.ckpt, args.config)
    model = model.to(device)
    model.eval()
    print(f"  img_size: {model.img_size}")
    print(f"  num_classes: {model.num_classes}")

    print("Building test dataloader ...")
    loader = build_test_loader(args.config, args.data_path)
    print(f"  Test set: {len(loader.dataset)} images")

    print(f"Saving predictions to: {args.output_dir}")
    run_inference(model, loader, device, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
