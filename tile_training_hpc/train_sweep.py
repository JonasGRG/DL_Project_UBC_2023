# Imports
import yaml
import wandb
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data_module import ImageFolderClassificationModule
from model import CustomModel

# CONSTANTS
# Model
NUM_CLASSES = 5

# Training
NUM_WORKERS=16
NUM_EPOCHS = 15

# Model
NUM_CLASSES = 5
CLASS_WEIGHTS = torch.tensor([0.7140, 0.5517, 0.3039, 1.6141, 1.8163])
PRETRAINED = True

# Data
TRAIN_DIR = '../data/UBC_tile_224_500k/train'
VAL_DIR = '../data/UBC_tile_224_500k/validation'
SUBSET=True


# Load sweep config from yaml YAML file
with open('sweep_config.yaml', 'r') as file:
    sweep_config = yaml.safe_load(file)

def train():
    # Initialize a new wandb run
    with wandb.init() as run:
        # Get the sweep configuration for this run
        config = run.config

        # Create the DataModule and Model using the wandb configuration
        data_module = ImageFolderClassificationModule(config, train_dir=TRAIN_DIR, val_dir=VAL_DIR, subset=SUBSET, num_workers=NUM_WORKERS)
        model = CustomModel(config, pretrained=PRETRAINED, num_classes=NUM_CLASSES, weight=CLASS_WEIGHTS)

        # Set up the WandbLogger
        wandb_logger = WandbLogger()

        # Initialize the Trainer with the WandbLogger
        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            logger=wandb_logger,
            accelerator='auto'
        )

        # Start training
        trainer.fit(model, datamodule=data_module)

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="UBC_tile_classification_sweep")

# Start the sweep agent
wandb.agent(sweep_id, function=train)


