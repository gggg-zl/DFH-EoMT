import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ==============================
# 需要你自己根据情况修改的配置
# ==============================

# 单张待预测图片路径
IMAGE_PATH = r"camvid/images/0001TP_008550.png"  # TODO: 改成你的图片路径

# 模型权重路径（如果有）
CHECKPOINT_PATH = r"checkpoints/final_model-spa_wet0.8rual.ckpt"  # TODO: 改成你的模型权重路径

# 配置文件路径（用于获取模型结构参数）
CONFIG_PATH = r"configs/dinov2/camvid/semantic/eomt_large_720.yaml"

# CamVid 11 类别的颜色表（RGB）
CLASS_NAMES = [
    "Sky", "Building", "Pole", "Road", "Pavement", "Tree",
    "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"
]


CLASS_COLORS = [
    (128, 128, 128),  # Sky         - 灰色
    (0,   0,   0  ),  # Building    - 黑色
    (128, 64,  0  ),  # Pole        - 棕色
    (128, 128, 0  ),  # Road        - 暗黄
    (64,  64,  64 ),  # Pavement    - 深灰
    (0,   128,  0 ),  # Tree        - 深绿
    (192, 0,   128),  # SignSymbol  - 紫色
    (128, 64,  128),  # Fence       - 浅紫
    (64,  0,   128),  # Car         - 深蓝紫
    (192, 128, 128),  # Pedestrian  - 浅粉
    (0,   64,   64 ),  # Bicyclist   - 深青
]

# 输出图片保存目录
OUTPUT_DIR = "camvid_pred"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# 加载模型（使用与训练一致的配置）
# ==============================

def load_model(ckpt_path: str, config_path: str):
    """
    加载 MaskClassificationSemantic 模型，结构和权重与训练时完全一致。
    """
    import yaml
    import importlib

    # 检查 checkpoint 是否存在
    if ckpt_path and not os.path.isfile(ckpt_path):
        print(f"WARNING: Checkpoint not found: {ckpt_path}")
        print("         Will continue with randomly initialized weights.")
        ckpt_path = None

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg  = cfg.get("data", {}).get("init_args", {})
    num_classes = data_cfg.get("num_classes", 11)
    img_size    = data_cfg.get("img_size", [480, 360])

    model_init = model_cfg["init_args"]
    model_init["num_classes"] = num_classes

    net_init = model_init["network"]["init_args"]
    net_init["num_classes"] = num_classes

    encoder_spec = net_init.get("encoder")
    if isinstance(encoder_spec, dict):
        if "init_args" not in encoder_spec:
            encoder_spec["init_args"] = {}
        encoder_spec["init_args"]["img_size"]   = tuple(img_size)
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

    from training.mask_classification_semantic import MaskClassificationSemantic
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


# ==============================
# 图片预处理
# ==============================

def load_image(image_path):
    """加载图片，返回 PIL Image"""
    img = Image.open(image_path).convert("RGB")
    return img


def resize_for_model(img_pil, target_h, target_w):
    """
    将 PIL Image resize 到 (target_h, target_w)，保持短边匹配，长边等比缩放后居中裁剪/填充。
    模型期望的尺寸: (target_h, target_w)，例如 (360, 480)。
    返回: result(PIL), orig_size, roi_info (paste_x, paste_y, new_w, new_h)
    """
    orig_w, orig_h = img_pil.size  # PIL: (W, H)

    # 计算缩放比例，使短边匹配目标
    scale = min(target_h / orig_h, target_w / orig_w)
    new_h, new_w = round(orig_h * scale), round(orig_w * scale)

    # Resize
    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)

    # 居中裁剪或填充到目标尺寸
    result = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    result.paste(img_resized, (paste_x, paste_y))

    return result, (orig_h, orig_w), (paste_x, paste_y, new_w, new_h)


# ==============================
# 推理（直接 resize 到模型窗口尺寸）
# ==============================

def predict_image(model, img_pil, device):
    """
    对单张图片进行语义分割预测。
    流程：resize 到模型窗口尺寸 -> 前向推理 -> 取 ROI 区域 -> 双线性插值回原尺寸。
    返回: numpy array, shape=(H, W), 每个像素为类别ID (0-10)
    """
    target_h, target_w = model.img_size  # e.g., (360, 480)
    orig_h, orig_w = img_pil.height, img_pil.width

    # resize + 居中裁剪/填充
    img_resized, orig_size, roi_info = resize_for_model(img_pil, target_h, target_w)
    paste_x, paste_y, new_w, new_h = roi_info

    # 转成 tensor [C, H, W], uint8
    img_np = np.array(img_resized)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

    # 前向推理
    with torch.no_grad():
        mask_logits_all, class_logits_all = model(img_tensor.unsqueeze(0))

        # 取最后一个 decoder block 的输出
        mask_logits  = mask_logits_all[-1][0]    # [num_q, h', w'] 低分辨率
        class_logits = class_logits_all[-1][0]   # [num_q, num_classes]

        # 必须与训练/验证一致：先把 mask 上采样到模型输入尺寸，再合成每像素 logit
        # （否则 crop_logits 仍是低分辨率，下面用 paste_x/y 按全分辨率切片会错位 → 竖条/花屏）
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0]

        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits.unsqueeze(0), class_logits.unsqueeze(0)
        )[0]  # [num_classes, target_h, target_w]

        # 只取 ROI 区域（去除黑边填充）
        roi_logits = crop_logits[:, paste_y:paste_y + new_h, paste_x:paste_x + new_w]

        # 双线性插值回原始尺寸
        pred_logits = F.interpolate(
            roi_logits.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )[0]  # [num_classes, orig_H, orig_W]

    # argmax 得到每个像素的类别
    pred_class = pred_logits.argmax(dim=0).cpu().numpy()  # (H, W)

    return pred_class


# ==============================
# 可视化函数
# ==============================

def colorize_prediction(pred, class_names, class_colors):
    """将类别ID图转换为彩色RGB图"""
    color_map = np.array(class_colors, dtype=np.uint8)
    rgb = color_map[pred]
    return rgb


def create_legend(class_names, class_colors):
    """创建类别图例"""
    patches = [
        mpatches.Patch(color=np.array(c) / 255.0, label=name)
        for name, c in zip(class_names, class_colors)
    ]
    return patches


# ==============================
# 主流程
# ==============================

def _next_index(base):
    i = 1
    while os.path.exists(base.format(i)):
        i += 1
    return i


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    print("Loading model ...")
    model = load_model(CHECKPOINT_PATH, CONFIG_PATH)
    model = model.to(device)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Model window size (H, W): {model.img_size}")
    print(f"Num classes: {model.num_classes}")

    # 2. 加载图片
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    img_pil = load_image(IMAGE_PATH)
    print(f"Input image: {IMAGE_PATH}, size: {img_pil.size}")

    # 3. 预测
    print("Predicting ...")
    pred = predict_image(model, img_pil, device)
    print(f"Prediction shape: {pred.shape}, unique classes: {np.unique(pred)}")

    # 4. 保存彩色分割图（自动递增序号，不覆盖已有文件）
    rgb_pred = colorize_prediction(pred, CLASS_NAMES, CLASS_COLORS)
    rgb_pil = Image.fromarray(rgb_pred)
    idx = _next_index(os.path.join(OUTPUT_DIR, "prediction_{:03d}.png"))
    save_path = os.path.join(OUTPUT_DIR, f"prediction_{idx:03d}.png")
    rgb_pil.save(save_path)
    print(f"Saved prediction to {save_path}")

    # 5. 可视化对比（原图 vs 分割结果）
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(rgb_pred)
    axes[1].set_title("CamVid Segmentation (11 classes)", fontsize=14)
    axes[1].axis("off")

    # 添加图例
    legend = create_legend(CLASS_NAMES, CLASS_COLORS)
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=min(6, len(CLASS_NAMES)),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )

    plt.tight_layout()
    compare_path = os.path.join(OUTPUT_DIR, f"compare_{idx:03d}.png")
    plt.savefig(compare_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison to {compare_path}")

    print("Done.")


if __name__ == "__main__":
    main()
