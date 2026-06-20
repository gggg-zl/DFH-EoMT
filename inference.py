# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# ---------------------------------------------------------------
# 推理脚本：使用训练好的权重对图片进行推理，支持单张和目录批量

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

from models.eomt import EoMT
from models.vit import ViT
from training.mask_classification_semantic import MaskClassificationSemantic


# trainId -> labelId 映射 (Cityscapes 官方提交格式)
TRAINID_TO_LABELID = {
    0: 7,    # road
    1: 8,    # sidewalk
    2: 11,   # building
    3: 12,   # wall
    4: 13,   # fence
    5: 17,   # pole
    6: 19,   # traffic light
    7: 20,   # traffic sign
    8: 21,   # vegetation
    9: 22,   # terrain
    10: 23,  # sky
    11: 24,  # person
    12: 25,  # rider
    13: 26,  # car
    14: 27,  # truck
    15: 28,  # bus
    16: 31,  # train
    17: 32,  # motorcycle
    18: 33,  # bicycle
}


def create_model(config_path: str, ckpt_path: str, device: str = "cuda", num_classes: int = None):
    """根据配置文件创建模型并加载权重"""
    print("正在加载配置文件...")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']['init_args']
    network_config = model_config['network']['init_args']
    data_config = config['data']['init_args']

    _num_classes = num_classes if num_classes is not None else data_config.get('num_classes', 19)
    img_size = tuple(data_config['img_size'])

    encoder_config_raw = network_config['encoder']
    encoder_config = dict(encoder_config_raw.get('init_args', {}))
    encoder_config['img_size'] = img_size

    print("正在创建模型...")
    encoder_class_path = encoder_config_raw.get('class_path', 'models.vit.ViT')
    if '.' in encoder_class_path:
        module_name, class_name = encoder_class_path.rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_name)
        encoder_cls = getattr(module, class_name)
    else:
        encoder_cls = ViT

    encoder_kwargs = {k: v for k, v in encoder_config.items() if k != 'class_path'}
    encoder = encoder_cls(**encoder_kwargs)

    network = EoMT(
        encoder=encoder,
        num_classes=_num_classes,
        num_q=network_config['num_q'],
        num_blocks=network_config.get('num_blocks', 4),
        masked_attn_enabled=network_config.get('masked_attn_enabled', True),
    )

    model = MaskClassificationSemantic(
        network=network,
        img_size=img_size,
        num_classes=_num_classes,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=None,
        attn_mask_annealing_end_steps=None,
        load_ckpt_class_head=False,
        ckpt_path=ckpt_path,
    )

    print(f"正在将模型移到 {device}...")
    model = model.to(device)
    model.eval()

    return model, config


# labelId -> RGB，与 visualization.py/Cityscapes 官方一致
LABELID_TO_COLOR = {
    7:  (128,  64, 128),   # road
    8:  (244,  35, 232),   # sidewalk
    11: ( 70,  70,  70),   # building
    12: (102, 102, 156),   # wall
    13: (190, 153, 153),   # fence
    17: (153, 153, 153),   # pole
    19: (250, 170,  30),   # traffic light
    20: (220, 220,   0),   # traffic sign
    21: (107, 142,  35),   # vegetation
    22: (152, 251, 152),   # terrain
    23: ( 70, 130, 180),   # sky
    24: (220,  20,  60),   # person
    25: (255,   0,   0),   # rider
    26: (  0,   0, 142),   # car
    27: (  0,   0,  70),   # truck
    28: (  0,  60, 100),   # bus
    31: (  0,  80, 100),   # train
    32: (  0,   0, 230),   # motorcycle
    33: (119,  11,  32),   # bicycle
}


def convert_trainid_to_color(pred_mask):
    """将模型输出的 mask (trainId 0-18) 转换为 RGB 彩色图，与 visualization.py 一致"""
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for train_id, label_id in TRAINID_TO_LABELID.items():
        color_mask[pred_mask == train_id] = LABELID_TO_COLOR[label_id]
    color_mask[pred_mask == 255] = (0, 0, 0)
    return color_mask


def convert_trainid_to_labelid(pred_mask):
    """将模型输出的 mask (trainId 0-18) 转换为官方提交所需的 mask"""
    submit_mask = np.full_like(pred_mask, 255, dtype=np.uint8)
    for train_id, label_id in TRAINID_TO_LABELID.items():
        submit_mask[pred_mask == train_id] = label_id
    return submit_mask


def predict_single_image(
    model,
    img_path: Path,
    img_size: tuple,
    device: str,
):
    """对单张图片进行推理，输入 uint8 [0,255]，输出 (H, W) 的 trainId 数组"""
    img_np = np.array(Image.open(img_path).convert('RGB'))
    orig_h, orig_w = img_np.shape[:2]

    # forward() 内部会 /255.0，所以传 uint8 原始值
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        crops, origins = model.window_imgs_semantic([img_tensor])
        mask_logits_all, class_logits_all = model(crops)

        mask_logits = F.interpolate(
            mask_logits_all[-1], model.img_size, mode='bilinear'
        )
        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_all[-1]
        )

        # revert_window_logits_semantic 内部已经 resize 回 orig 尺寸
        logits_list = model.revert_window_logits_semantic(
            crop_logits, origins, [(orig_h, orig_w)]
        )
        pred_logits = logits_list[0]  # (C, H, W)

        # argmax 得到 (H, W)，转 numpy
        preds = pred_logits.argmax(dim=0).cpu().numpy().astype(np.uint8)

    # 确保是 2D 数组
    preds = np.reshape(preds, (orig_h, orig_w))
    return preds


def run_inference(
    config_path: str,
    ckpt_path: str,
    input_path: str,
    output_path: str = None,
    device: str = "cuda",
    color_output: bool = False,
):
    """运行推理，支持单张图片或目录"""
    print(f"=" * 50)
    print(f"配置文件: {config_path}")
    print(f"权重文件: {ckpt_path}")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    print(f"设备: {device}")
    print(f"彩色输出: {'是' if color_output else '否'}")
    print(f"=" * 50)

    model, config = create_model(config_path, ckpt_path, device)
    num_classes = config['data']['init_args'].get('num_classes', 19)
    img_size = tuple(config['data']['init_args']['img_size'])
    print(f"模型加载完成! 类别数: {num_classes}, 模型输入尺寸: {img_size}")

    input_path = Path(input_path)

    # ---------- 单张图片 ----------
    if input_path.is_file() and input_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
        print(f"检测到单张图片: {input_path.name}")

        pred_mask = predict_single_image(model, input_path, img_size, device)

        if output_path:
            out_path = Path(output_path)
        else:
            stem = input_path.stem
            out_path = input_path.parent / f"{stem}_pred.png"

        os.makedirs(out_path.parent, exist_ok=True)

        if color_output:
            color_mask = convert_trainid_to_color(pred_mask)
            Image.fromarray(color_mask, mode='RGB').save(str(out_path))
        else:
            submit_mask = convert_trainid_to_labelid(pred_mask)
            Image.fromarray(submit_mask, mode='L').save(str(out_path))

        print(f"完成! 结果保存在: {out_path}")

    # ---------- 目录批量 ----------
    elif input_path.is_dir():
        print(f"检测到目录: {input_path}")

        if output_path:
            out_dir = Path(output_path)
        else:
            out_dir = input_path.parent / f"{input_path.name}_pred"
        os.makedirs(out_dir, exist_ok=True)

        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([f for f in input_path.rglob('*') if f.is_file() and f.suffix.lower() in exts])

        print(f"找到 {len(image_files)} 张图片")

        if len(image_files) == 0:
            print("未找到任何图片!")
            return

        for img_path in tqdm(image_files, desc="推理中"):
            try:
                pred_mask = predict_single_image(model, img_path, img_size, device)

                if '_leftImg8bit' in img_path.name:
                    out_name = img_path.name.replace('_leftImg8bit', '')
                else:
                    out_name = img_path.name

                if not out_name.endswith('.png'):
                    out_name = Path(out_name).stem + '.png'

                if color_output:
                    color_mask = convert_trainid_to_color(pred_mask)
                    Image.fromarray(color_mask, mode='RGB').save(str(out_dir / out_name))
                else:
                    submit_mask = convert_trainid_to_labelid(pred_mask)
                    Image.fromarray(submit_mask, mode='L').save(str(out_dir / out_name))

            except Exception as e:
                print(f"处理 {img_path.name} 出错: {e}")
                continue

        print(f"\n完成! 结果保存在: {out_dir}")

    else:
        print(f"无效的输入路径: {input_path}")
        print("请提供单张图片文件或包含图片的目录")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EoMT 推理脚本（支持单张和目录）')
    parser.add_argument('--config', type=str, required=True, help='配置文件')
    parser.add_argument('--checkpoint', type=str, required=True, help='权重文件')
    parser.add_argument('--input', type=str, required=True,
                        help='输入路径：单张图片文件 或 图片目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径：单张时为文件名，批量时为目录（默认: 同目录 _pred/）')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--color', action='store_true', help='输出 RGB 彩色分割图')

    args = parser.parse_args()

    run_inference(
        config_path=args.config,
        ckpt_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        color_output=args.color,
    )
