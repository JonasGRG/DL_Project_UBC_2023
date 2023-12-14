import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassConfusionMatrix
import timm

from losses import FocalLoss


class CustomModel(pl.LightningModule):
    def __init__(self, config, pretrained=True, num_classes=5, weight=None):
        super().__init__()
        self.config = config
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model = self.create_model()
        self.criterion = self.set_criterion(weight)
        self.mcm = MulticlassConfusionMatrix(num_classes)


    def set_criterion(self, weight):
        if self.config['loss'] == 'focal':
            return FocalLoss(weight)
        elif self.config['loss'] == 'cross_entropy':
            return nn.CrossEntropyLoss(weight=weight)

    def create_model(self):
        # Create the model using timm
        model = timm.create_model(
            self.config['model_name'],
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            drop_rate = self.config['drop_rate']
            )
        return model


    def balanced_accuracy(self, y_true, y_pred):
        C = self.mcm(y_pred, y_true)
        per_class = torch.diag(C) / C.sum(axis=1)
        if torch.any(torch.isnan(per_class)):
            per_class = per_class[~torch.isnan(per_class)]
        score = per_class.mean()

        return score


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        balanced_acc = self.balanced_accuracy(y, preds)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_balanced_accuracy', balanced_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        balanced_acc = self.balanced_accuracy(y, preds)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_balanced_accuracy', balanced_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        # Separate parameters for the backbone and the head
        backbone_parameters = [p for name, p in self.model.named_parameters() if not name.startswith('head')]
        head_parameters = [p for name, p in self.model.named_parameters() if name.startswith('head')]

        # Learning rates from config
        backbone_lr = self.config['backbone_lr']
        head_lr = self.config['head_lr']

        # Optimizer selection
        optimizer_type = self.config['optimizer']
        weight_decay = self.config['weight_decay']
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam([
                {'params': backbone_parameters, 'lr': backbone_lr},
                {'params': head_parameters, 'lr': head_lr}
            ], weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW([
                {'params': backbone_parameters, 'lr': backbone_lr},
                {'params': head_parameters, 'lr': head_lr}
            ], weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        scheduler_type = self.config['scheduler']
        if scheduler_type == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        else:
            return optimizer

        return {"optimizer": optimizer, "lr_scheduler": scheduler}