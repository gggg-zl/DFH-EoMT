# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.datasets import Cityscapes

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
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


class CityscapesSemantic(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 8,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
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
        masks, labels = [], []

        for label_id in target[0].unique():
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )
        self.cityscapes_val_dataset = Dataset(
            transforms=self.val_transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )
        self.cityscapes_test_dataset = Dataset(
            transforms=self.val_transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/test"),
            target_folder_path_in_zip=Path("./gtFine/test"),
            **cityscapes_dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cityscapes_test_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
