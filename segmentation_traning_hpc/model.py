import pytorch_lightning as pl
import torch
import monai
from monai.losses import MaskedDiceLoss
from monai.networks.utils import one_hot

class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.5
        )
        self.loss_fn = MaskedDiceLoss(include_background=True)
        self.lr = 3e-4
        self.weight_decay = 1e-6
        self.T_max = 30

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=0)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image, label, mask = batch['img'], batch['label'], batch['mask']
        label = one_hot(label, num_classes=3)[:, 1:]
        pred = self(image)
        loss = self.loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label, mask = batch['img'], batch['label'], batch['mask']
        label = one_hot(label, num_classes=3)[:, 1:]
        pred = self(image)
        loss = self.loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)
        return loss
