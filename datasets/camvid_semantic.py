# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union, Optional
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import tv_tensors

from torchvision.transforms import v2 as T

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset as ZipDataset
from datasets.transforms import Transforms


class ValTransforms:
    """验证/测试集变换：仅将图片和掩码 resize 到指定尺寸"""

    def __init__(self, img_size: tuple[int, int]):
        h, w = img_size[1], img_size[0]
        self.img_resize = T.Resize((h, w), antialias=True)
        self.mask_resize = T.Resize((h, w), interpolation=T.InterpolationMode.NEAREST)

    def __call__(self, img, target):
        img = self.img_resize(img)
        target["masks"] = self.mask_resize(target["masks"])
        return img, target


# CamVid 类别定义 (11类)
class CamVidClasses:
    # 类别ID到名称的映射
    classes = [
        "Sky", "Building", "Pole", "Road", "Pavement", "Tree",
        "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist"
    ]
    
    # 原始32类ID到新11类ID的映射
    ORIGINAL_TO_TRAIN_ID = {
        21: 0,  # Sky
        4: 1,   # Building
        8: 2,   # Column_Pole -> Pole
        17: 3,  # Road
        19: 4,  # Sidewalk -> Pavement
        26: 5,  # Tree
        20: 6,  # SignSymbol
        9: 7,   # Fence
        5: 8,   # Car
        16: 9,  # Pedestrian
        2: 10,  # Bicyclist
        255: 255,  # Void
    }
    
    @staticmethod
    def get_color_to_class_map():
        """获取颜色到类别的映射"""
        # CamVid标签是PNG格式，像素值即为类别ID (0-32)
        return None


class CamVidSemantic(LightningDataModule):
    """
    CamVid 语义分割数据集
    数据目录结构:
        path/
            images/
                0001TP_006690.png
                ...
            labels/
                0001TP_006690.png
                ...
    """
    
   
    NUM_CLASSES = 11
    
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 8,
        img_size: tuple[int, int] = (480,360),  # 处理分辨率，原始 720×960 的一半
        num_classes: int = 11,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        ignore_idx: int = 255,  # CamVid的Void类别
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=ignore_idx,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        self.val_transforms = ValTransforms(img_size=img_size)

    @staticmethod
    def target_parser(target, **kwargs):
        """
        解析语义分割标签
        将原始32类ID映射到新的11类ID
        """
        masks, labels, is_crowd = [], [], []
        
        # 获取所有唯一的类别ID（排除255 Void类别）
        unique_labels = target[0].unique()
        
        for label_id in unique_labels:
            label_id_int = int(label_id)
            if label_id_int == 255:  # 忽略Void类别
                continue
            
            # 映射到新的11类ID
            new_label_id = CamVidClasses.ORIGINAL_TO_TRAIN_ID.get(label_id_int)
            if new_label_id is None:
                continue  # 忽略未映射的类别
                
            masks.append(target[0] == label_id)
            labels.append(int(new_label_id))
            is_crowd.append(False)
        
        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        """加载训练集、验证集和测试集"""
        path = Path(self.path)
        
        # 验证集 (valid.txt)
        self.val_dataset = CamVidFolderDataset(
            img_dir=path / "images",
            label_dir=path / "labels",
            transforms=self.val_transforms,
            target_parser=self.target_parser,
            check_empty_targets=self.check_empty_targets,
            ignore_idx=self.ignore_idx,
            split_file=path / "valid.txt",  # 验证集文件列表
        )
        
        # 测试集 (test.txt)
        self.test_dataset = CamVidFolderDataset(
            img_dir=path / "images",
            label_dir=path / "labels",
            transforms=self.val_transforms,
            target_parser=self.target_parser,
            check_empty_targets=False,  # 测试集不检查空标签
            ignore_idx=self.ignore_idx,
            split_file=path / "test.txt",  # 测试集文件列表
        )
        
        # 训练集 (train.txt)
        self.train_dataset = CamVidFolderDataset(
            img_dir=path / "images",
            label_dir=path / "labels",
            transforms=self.transforms,
            target_parser=self.target_parser,
            check_empty_targets=self.check_empty_targets,
            ignore_idx=self.ignore_idx,
            split_file=path / "train.txt",  # 训练集文件列表
        )
        
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


class CamVidFolderDataset(torch.utils.data.Dataset):
    """
    从文件夹加载CamVid数据集
    """
    
    def __init__(
        self,
        img_dir: Path,
        label_dir: Path,
        transforms: Optional = None,
        target_parser=None,
        check_empty_targets: bool = True,
        ignore_idx: int = 255,
        split_file: Optional[Path] = None,  # 用于指定验证集的文件列表
        exclude_split_file: Optional[Path] = None,  # 用于排除的文件列表（训练集使用）
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.target_parser = target_parser
        self.check_empty_targets = check_empty_targets
        self.ignore_idx = ignore_idx
        
        # 读取需要排除/包含的文件列表
        exclude_files = set()
        if exclude_split_file and exclude_split_file.exists():
            with open(exclude_split_file, 'r') as f:
                exclude_files = set(line.strip() for line in f.readlines())
        
        if split_file and split_file.exists():
            # 只加载指定列表中的文件
            with open(split_file, 'r') as f:
                valid_files = [line.strip() for line in f.readlines()]
            
            # 过滤只保留存在的文件（图片和标签都必须存在）
            self.img_files = []
            self.label_files = []
            for fname in valid_files:
                img_path = self.img_dir / fname
                label_path = self.label_dir / fname
                if img_path.exists() and label_path.exists():
                    self.img_files.append(img_path)
                    self.label_files.append(label_path)
        else:
            # 加载所有图片文件，排除指定文件
            all_img_files = sorted(list(self.img_dir.glob("*.png")))
            self.img_files = []
            self.label_files = []
            for f in all_img_files:
                if f.name not in exclude_files:
                    self.img_files.append(f)
                    self.label_files.append(self.label_dir / f.name)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx: int):
        # 加载图片
        img = Image.open(self.img_files[idx]).convert("RGB")
        img = tv_tensors.Image(img)
        
        # 加载标签
        label = Image.open(self.label_files[idx])
        label = tv_tensors.Mask(label, dtype=torch.long)
        
        # 确保图片和标签尺寸一致
        if img.shape[-2:] != label.shape[-2:]:
            from torchvision.transforms.v2 import functional as F
            label = F.resize(
                label,
                list(img.shape[-2:]),
                interpolation=F.InterpolationMode.NEAREST,
            )
        
        # 解析标签
        masks, labels, is_crowd = self.target_parser(
            target=label,
            stuff_classes=None,
        )
        
        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)) if masks else tv_tensors.Mask(torch.zeros(0, *label.shape[-2:], dtype=torch.long)),
            "labels": torch.tensor(labels) if labels else torch.tensor([], dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd) if is_crowd else torch.tensor([], dtype=torch.bool),
        }
        
        # 应用变换
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
