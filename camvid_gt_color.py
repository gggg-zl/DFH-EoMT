# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# ---------------------------------------------------------------
# CamVid GT 标签转彩色可视化脚本（单张图片版）
#
# 将调色板模式 PNG 的标签转为 RGB 彩色图，可选：
#   1. 直接保存彩色 GT 图
#   2. 将彩色 GT 叠加在原图上生成 overlay
#
# 用法:
#   python camvid_gt_color.py --label camvid/labels/0001TP_006690_P.png
#   python camvid_gt_color.py --label camvid/labels/0001TP_006690_P.png --mode overlay
#   python camvid_gt_color.py --label camvid/labels/0001TP_006690_P.png --mode both --output result.png
# ---------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
from PIL import Image


# ============================================================
# CamVid 原始类别 ID -> 训练 ID (0-10) 映射
# 来源: datasets/camvid_semantic.py
# ============================================================
CAMVID_ORIGINAL_TO_TRAIN_ID = {
    21: 0,  # Sky
     4: 1,  # Building
     8: 2,  # Column_Pole -> Pole
    17: 3,  # Road
    19: 4,  # Sidewalk -> Pavement
    26: 5,  # Tree
    20: 6,  # SignSymbol
     9: 7,  # Fence
     5: 8,  # Car
    16: 9,  # Pedestrian
     2: 10, # Bicyclist
}


# ============================================================
# CamVid 11 类颜色调色板 (RGB)
# palette 中每个索引对应的 RGB 颜色
# ============================================================
CAMVID_CLASS_COLORS = [
    (128, 128, 128),  # Sky
    (128,   0,   0),  # Building
    (192, 192, 128),  # Pole
    (128, 128,   0),  # Road
    (  0,   0, 192),  # Pavement
    (128, 128, 128),  # Tree
    (192, 128, 128),  # SignSymbol
    (128,  64, 128),  # Fence
    (  0,   0, 128),  # Car
    ( 64,  64, 128),  # Pedestrian
    ( 64,   0, 128),  # Bicyclist
]


def _build_original_to_color_map(class_colors: list) -> dict:
    """
    构建 原始像素值 -> RGB 颜色的完整查找表。
    像素值 = CamVid 原始类别 ID，映射到 train ID，再查颜色表。
    """
    orig_to_rgb = {}
    for orig_id, train_id in CAMVID_ORIGINAL_TO_TRAIN_ID.items():
        orig_to_rgb[orig_id] = class_colors[train_id]
    return orig_to_rgb


ORIGINAL_ID_TO_RGB = _build_original_to_color_map(CAMVID_CLASS_COLORS)


def load_label_as_rgb(label_path: Path) -> np.ndarray:
    """
    读取 L 模式标签 PNG。
    像素值是 CamVid 原始类别 ID（21=Sky, 4=Building, 5=Car 等），
    先映射到 train ID (0-10)，再查调色板还原 RGB 彩色图。
    """
    pil_img = Image.open(label_path)
    arr = np.array(pil_img, dtype=np.uint8)

    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for orig_id, color in ORIGINAL_ID_TO_RGB.items():
        mask = arr == orig_id
        rgb[mask] = color

    return rgb


def overlay_on_image(rgb: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将彩色 mask 叠加在原图上。"""
    blended = (rgb.astype(np.float32) * (1 - alpha) + colored_mask.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_single(
    label_path: str | Path,
    mode: str = "colored",
    alpha: float = 0.5,
    output_path: str | Path = None,
):
    """
    处理单张 CamVid GT 标签图（调色板模式 PNG）。

    Args:
        label_path: GT 标签 PNG 路径
        mode: "colored" | "overlay" | "both"
        alpha: overlay 模式的透明度 (0.0 - 1.0)
        output_path: 输出路径
            - mode=colored: 默认保存为 <label>_colored.png
            - mode=overlay: 默认保存为 <label>_overlay.png
            - mode=both: 默认保存为 <label>_colored.png 和 <label>_overlay.png
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    stem = label_path.stem
    parent = Path(".")

    # 读取调色板模式 PNG，直接得到 RGB 彩色图
    colored = load_label_as_rgb(label_path)

    # 对应原图路径（与 labels 同级的 images 目录）
    # 标签文件名可能带 _P 后缀（如 0001TP_006690_P.png），原图则没有（如 0001TP_006690.png）
    img_dir = parent.parent / "images" if parent.name == "labels" else parent / "images"
    img_stem = stem.removesuffix("_P") if stem.endswith("_P") else stem
    img_path = img_dir / f"{img_stem}.png"

    if mode == "colored":
        out_path = Path(output_path) if output_path else parent / f"{stem}_colored.png"
        Image.fromarray(colored).save(out_path)
        print(f"Saved colored GT: {out_path}")

    elif mode == "overlay":
        if not img_path.exists():
            raise FileNotFoundError(f"Original image not found: {img_path}")
        rgb = np.array(Image.open(img_path).convert("RGB"))
        blended = overlay_on_image(rgb, colored, alpha=alpha)
        out_path = Path(output_path) if output_path else parent / f"{stem}_overlay.png"
        Image.fromarray(blended).save(out_path)
        print(f"Saved overlay: {out_path}")

    elif mode == "both":
        out_colored = Path(output_path) if output_path else parent / f"{stem}_colored.png"
        Image.fromarray(colored).save(out_colored)
        print(f"Saved colored GT: {out_colored}")

        if not img_path.exists():
            raise FileNotFoundError(f"Original image not found: {img_path}")
        rgb = np.array(Image.open(img_path).convert("RGB"))
        blended = overlay_on_image(rgb, colored, alpha=alpha)
        out_overlay = parent / f"{stem}_overlay.png"
        Image.fromarray(blended).save(out_overlay)
        print(f"Saved overlay: {out_overlay}")


def process_batch(
    labels_dir: str | Path,
    output_dir: str | Path = None,
    mode: str = "colored",
    alpha: float = 0.5,
):
    """
    批量处理 CamVid GT 标签目录。

    Args:
        labels_dir: GT 标签 PNG 目录
        output_dir: 输出目录（默认为 labels 同级的 labels_colored）
        mode: "colored" | "overlay" | "both"
        alpha: overlay 模式的透明度
    """
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    if output_dir is None:
        output_dir = labels_dir.parent / f"{labels_dir.name}_colored"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 对应原图目录
    img_dir = labels_dir.parent / "images"

    label_files = sorted(labels_dir.glob("*.png"))
    if not label_files:
        raise FileNotFoundError(f"No PNG label files found in: {labels_dir}")

    print(f"Found {len(label_files)} label files. Processing...")

    for i, label_path in enumerate(label_files, 1):
        stem = label_path.stem
        # 标签文件名可能带 _P 后缀，原图则没有
        img_stem = stem.removesuffix("_P") if stem.endswith("_P") else stem

        # 读取调色板模式 PNG，直接得到 RGB 彩色图
        colored = load_label_as_rgb(label_path)

        if mode == "colored":
            out_path = output_dir / f"{stem}_colored.png"
            Image.fromarray(colored).save(out_path)

        elif mode == "overlay":
            img_path = img_dir / f"{img_stem}.png"
            if not img_path.exists():
                print(f"  [Skip] Original image not found: {img_path}")
                continue
            rgb = np.array(Image.open(img_path).convert("RGB"))
            blended = overlay_on_image(rgb, colored, alpha=alpha)
            out_path = output_dir / f"{stem}_overlay.png"
            Image.fromarray(blended).save(out_path)

        elif mode == "both":
            out_colored = output_dir / f"{stem}_colored.png"
            Image.fromarray(colored).save(out_colored)

            img_path = img_dir / f"{img_stem}.png"
            if not img_path.exists():
                print(f"  [Skip] Original image not found: {img_path}")
                continue
            rgb = np.array(Image.open(img_path).convert("RGB"))
            blended = overlay_on_image(rgb, colored, alpha=alpha)
            out_overlay = output_dir / f"{stem}_overlay.png"
            Image.fromarray(blended).save(out_overlay)

        if i % 50 == 0 or i == len(label_files):
            print(f"  [{i}/{len(label_files)}] Processed: {stem}")

    print(f"\nBatch processing complete. Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert CamVid GT labels to colored visualization")
    parser.add_argument(
        "--label",
        type=str,
        help="Path to a single CamVid GT label PNG file, or a directory to batch process"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        help="Directory containing CamVid GT label PNG files (alternative to --label for batch mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch mode (default: <labels_dir>_colored)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["colored", "overlay", "both"],
        default="colored",
        help="'colored': save RGB GT image; 'overlay': GT on original image; 'both': both"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency of GT overlay on original image (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (for single-file mode=colored or overlay)"
    )
    args = parser.parse_args()

    # 优先使用 --labels-dir 进行批量处理
    if args.labels_dir:
        process_batch(
            labels_dir=args.labels_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            alpha=args.alpha,
        )
    elif args.label:
        process_single(
            label_path=args.label,
            mode=args.mode,
            alpha=args.alpha,
            output_path=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
