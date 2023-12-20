import os
import re
from glob import glob
from typing import Dict, Optional
from time import time
import warnings
import yaml

from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.data import MetaTensor

from monai.transforms import (Compose,CopyItemsd, LabelToMaskd,
                              Lambdad, RandRotate90d, ScaleIntensityd,
                              Rand2DElasticd, RandFlipd, RandGaussianNoised)


from PIL import Image
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning import Trainer

import utils
import data_module
from model import SegmentationModel

NUM_EPOCHS = 20


# Ign$ore specific deprecation warnings related to TypedStorage
warnings.filterwarnings('ignore', category=UserWarning, message='.*TypedStorage is deprecated.*')

Image.MAX_IMAGE_PIXELS = None

train_files, val_files = utils.get_train_val_files()

train_loader, val_loader = data_module.get_dataloaders(train_files, val_files)

model = SegmentationModel()
trainer = Trainer(max_epochs=NUM_EPOCHS, logger = )
trainer.fit(model, train_loader, val_loader)



