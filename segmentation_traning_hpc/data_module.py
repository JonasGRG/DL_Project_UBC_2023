import numpy as np
from torch.utils.data import DataLoader

import monai
from monai.data import MetaTensor
from monai.transforms import (Compose,CopyItemsd, LabelToMaskd,
                              Lambdad, RandRotate90d, ScaleIntensityd,
                              Rand2DElasticd, RandFlipd, RandGaussianNoised)

from PIL import Image

from randstainad import RandStainNAd
import utils

YAML_FILE = 'data/CRC_LAB_randomTrue_n0.yaml'

# define transforms for image and segmentation
train_transforms = Compose(
    [
        Lambdad(keys=['img'], func=lambda x: Image.open(x)),
        Lambdad(keys=['label'], func=lambda x: np.load(x)),
        RandStainNAd(keys=["img"], yaml_file="/content/local_data/CRC_LAB_randomTrue_n0 (1).yaml", std_hyper=-0.15, distribution="normal", probability=0.9, is_train=True),
        Lambdad(keys=['img'], func=lambda img_np: MetaTensor(img_np) if isinstance(img_np, np.ndarray) else MetaTensor(np.array(img_np))),
        Lambdad(keys=['img'], func=lambda x: x.permute(2,0,1)),
        Lambdad(keys=['label'], func=lambda x: x.unsqueeze(0)),

        CopyItemsd(keys=['label'], times=1, names=['mask']),                                        # Copy label to new image mask
        LabelToMaskd(keys=['mask'], select_labels=[1,2], merge_channels=True, allow_missing_keys=True),
        Lambdad(keys=['img'], func=utils.change_background_color),
        ScaleIntensityd(keys=["img"]),
        monai.transforms.Resized(keys=['img','label', 'mask'], spatial_size=(1024, 1024)),
        RandRotate90d(keys=["img", "label", "mask"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["img", "label", "mask"], prob=0.3, spatial_axis=0),
        RandFlipd(keys=["img", "label", "mask"], prob=0.3, spatial_axis=1),
        Rand2DElasticd(keys=["img", "label", "mask"],
                       prob=0.5,
                       spacing=(30, 30),
                       magnitude_range=(0, 0.5),
                       rotate_range=(0.2, 0.2),
                       scale_range=(-0.05,0.05),
                       mode="bilinear",
                       padding_mode="reflection"),
        RandGaussianNoised(keys=["img"], prob=0.05, mean=0.0, std=0.1),

    ]
)

val_transforms = Compose(
    [
        Lambdad(keys=['img'], func=lambda x: Image.open(x)),
        Lambdad(keys=['label'], func=lambda x: np.load(x)),
        Lambdad(keys=['img'], func=lambda img_np: MetaTensor(img_np) if isinstance(img_np, np.ndarray) else MetaTensor(np.array(img_np))),
        Lambdad(keys=['img'], func=lambda x: x.permute(2,0,1)),
        Lambdad(keys=['label'], func=lambda x: x.unsqueeze(0)),

        CopyItemsd(keys=['label'], times=1, names=['mask']),                                        # Copy label to new image mask
        LabelToMaskd(keys=['mask'], select_labels=[1,2], merge_channels=True, allow_missing_keys=True),
        Lambdad(keys=['img'], func=utils.change_background_color),
        ScaleIntensityd(keys=["img"]),
        monai.transforms.Resized(keys=['img','label', 'mask'], spatial_size=(1024, 1024)),
    ]
)

def get_dataloaders(train_files, val_files):
    train_dataset = monai.data.Dataset(data=train_files, transform=train_transforms)
    val_dataset = monai.data.Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader