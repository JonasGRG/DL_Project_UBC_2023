import numpy as np
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision import datasets
from monai.transforms import (Compose,EnsureChannelFirst,
                              ScaleIntensity,RandZoom, RandRotate90,
                              Rand2DElastic, RandFlip)


from utils import RandStainNA

YAML_FILE = '../data/CRC_LAB_randomTrue_n0.yaml'


def light_train_transform():
    # Define light augmentation transforms
    return Compose([
        lambda x: np.array(x),
        EnsureChannelFirst(channel_dim=2),
        ScaleIntensity(),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.25, spatial_axis=1),
        RandFlip(prob=0.25, spatial_axis=0),
    ])

def moderate_train_transform():
    return Compose([
        RandStainNA(
            yaml_file=YAML_FILE,
            std_hyper=-0.15, distribution="normal",
            probability=0.8,
            is_train=True
            ),
        EnsureChannelFirst(channel_dim=2),
        ScaleIntensity(),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.25, spatial_axis=1),
        RandFlip(prob=0.25, spatial_axis=0),
        RandZoom(
            prob=0.1,
            min_zoom=0.8,
            max_zoom=1.2,
            mode = "bilinear",
            padding_mode='reflect'
            ) # Scale augmentation on WSI tile level

    ])

def heavy_train_transform():
    return Compose([
        RandStainNA(
            yaml_file=YAML_FILE,
            std_hyper=-0.3, distribution="normal",
            probability=0.95,
            is_train=True
            ),
        EnsureChannelFirst(channel_dim=2),
        ScaleIntensity(),
        Rand2DElastic(
            prob=0.5,
            spacing=(10,10),
            magnitude_range=(0.05, 0.5)
            ),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.25, spatial_axis=1),
        RandFlip(prob=0.25, spatial_axis=0),
        RandZoom(
            prob=0.1,
            min_zoom=0.8,
            max_zoom=2,
            mode = "bilinear",
            padding_mode='reflect'
            ), # Scale augmentation on WSI tile level
    ])

def val_transform():
  return Compose([
        lambda x: np.array(x),
        EnsureChannelFirst(channel_dim=2),
        ScaleIntensity(),
    ])

def stratified_subset(dataset, test_size=0.05, random_state=42):
    labels = [label for _, label in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for _, subset_indices in sss.split(dataset.samples, labels):
        return Subset(dataset, subset_indices)

class ImageFolderClassificationModule(pl.LightningDataModule):
    def __init__(self, config, train_dir, val_dir, subset=False, num_workers=4):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = num_workers
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.train_transform, self.val_transform = self._get_transform()
        self.subset = subset

    def _get_transform(self):
        # Choose the correct transform based on config
        if self.config['augmentation'] == 'light':
            return light_train_transform(), val_transform()
        elif self.config['augmentation'] == 'moderate':
            return moderate_train_transform(), val_transform()
        else:
            return heavy_train_transform(), val_transform()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        full_train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transform)
        full_val_dataset = datasets.ImageFolder(root=self.val_dir, transform=self.val_transform)

        if self.subset:
            self.train_dataset = stratified_subset(full_train_dataset)
            self.val_dataset = stratified_subset(full_val_dataset)
        else:
            self.train_dataset = full_train_dataset
            self.val_dataset = full_val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)