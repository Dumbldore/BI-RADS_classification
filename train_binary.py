import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
from bin.basic_datamodule import BasicDataModule
from bin.binary_dataset import TumorDataset
from bin.utils import MyRotateTransform
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from torch import Tensor
from torchvision import models

# Set seeds for reproducibility
seed_everything(42, workers=True)
torch.manual_seed(42)
np.random.seed(42)


class ClassifierLightning(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 6,
        lr: float = 0.0001,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="model")
        self._model: nn.Module = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )
        self._model.classifier[1] = nn.Linear(
            self._model.classifier[1].in_features, num_classes
        )
        self._train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._train_weighted_acc = torchmetrics.Accuracy(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self._train_f1_score = torchmetrics.F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self._val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_weighted_acc = torchmetrics.Accuracy(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self._val_f1_score = torchmetrics.F1Score(
            task="multiclass", average="macro", num_classes=num_classes
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self._model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = train_transforms(x)
        y_pred = self._model(x)
        y_pred = torch.squeeze(y_pred)
        loss = F.binary_cross_entropy_with_logits(
            y_pred, y.float(), label_smoothing=0.1
        )
        self.log("train_loss", loss, on_epoch=False, prog_bar=True)
        self._train_acc(y_pred, y)
        self.log("train_acc", self._train_acc, on_epoch=False, prog_bar=True)
        self._train_weighted_acc(y_pred, y)
        self.log(
            "train_weighted_acc",
            self._train_weighted_acc,
            on_epoch=False,
            prog_bar=True,
        )
        self._train_f1_score(y_pred, y)
        self.log("train_f1_score", self._train_f1_score, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self._model(x)
        y_pred = torch.squeeze(y_pred)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        self.log("val_loss", loss, prog_bar=True)
        self._val_acc(y_pred, y)
        self.log("val_acc", self._val_acc, prog_bar=True)
        self._val_weighted_acc(y_pred, y)
        self.log("val_weighted_acc", self._val_weighted_acc, prog_bar=True)
        self._val_f1_score(y_pred, y)
        self.log("val_f1_score", self._val_f1_score, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Use with model.eval() and 'with torch.no_grad():'.
        x, _ = batch
        return self(x.float())

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


transformations = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        MyRotateTransform(angles=[0, 180]),
    ]
)

checkpoint_name = "model_birads"
BATCH_SIZE = 36
DATASET_PATH = "/images"
dataset = ["BUS", "BUSI"]


model = ClassifierLightning()

tumor_datamodule = BasicDataModule(
    dataset_class=TumorDataset,
    dataset_init_kwargs=dict(
        dataset_path=DATASET_PATH, data_transforms=transformations, folders=dataset
    ),
    train_val_test_fractions=(0.8, 0.2, 0.0),
    batch_size=BATCH_SIZE,
    num_workers=4,
    kfold_splits=5,
    n_split_kfold=0,
)

early_stopping_callback = EarlyStopping(
    monitor="val_f1_score", patience=40, mode="max", verbose=True
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"checkpoints/{checkpoint_name}",
    save_top_k=1,
    monitor="val_weighted_acc",
    mode="max",
    auto_insert_metric_name=True,
    filename="{epoch}-{val_weighted_acc:.3f}-{val_f1_score:.3f}",
)

model.train()
trainer = Trainer(
    max_epochs=150,
    deterministic="True",
    callbacks=[
        early_stopping_callback,
        checkpoint_callback,
        StochasticWeightAveraging(swa_lrs=1e-2),
    ],
)
trainer.fit(model, datamodule=tumor_datamodule)
