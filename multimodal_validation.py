import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision.transforms as transforms
from pytorch_lightning import Trainer, seed_everything
from torch import Tensor
from torchvision import models

from data.basic_datamodule import BasicDataModule
from data.binary_dataset import MultimodalTumorDataset
from models.multimodal_model import MajorityVotingMultimodalModel

seed_everything(42, workers=True)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.set_float32_matmul_precision("medium")


class ClassifierLightning(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 1,
        hyperparameters_omitted_in_checkpoint="model",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=hyperparameters_omitted_in_checkpoint)
        self._model: nn.Module = model
        self._val_acc = torchmetrics.Accuracy(task="binary", num_classes=num_classes)
        self._val_weighted_acc = torchmetrics.Accuracy(
            task="binary", average="macro", num_classes=num_classes
        )
        self._val_f1_score = torchmetrics.F1Score(
            task="binary", average="macro", num_classes=num_classes
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self._model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self._model(x)
        self._val_acc(y_pred, y)
        self.log("val_acc", self._val_acc, prog_bar=True)
        self._val_weighted_acc(y_pred, y)
        self.log("val_weighted_acc", self._val_weighted_acc, prog_bar=True)
        self._val_f1_score(y_pred, y)
        self.log("val_f1_score", self._val_f1_score, prog_bar=True)
        return 1

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class MultimodalModelLightning(ClassifierLightning):
    def __init__(
        self,
        m1: nn.Module,
        m2: nn.Module,
        m3: nn.Module,
        multimodal_model,
        num_classes: int,
        hyperparameters_omitted_in_checkpoint,
    ) -> None:
        my_model = multimodal_model([m1, m2, m3], num_classes)

        super(MultimodalModelLightning, self).__init__(
            model=my_model,
            num_classes=num_classes,
            hyperparameters_omitted_in_checkpoint=hyperparameters_omitted_in_checkpoint,  # noqa: E501
        )


transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class EfficientLightning(ClassifierLightning):
    def __init__(self, num_classes: int = 1):
        model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        super().__init__(model, num_classes)


model_1 = EfficientLightning.load_from_checkpoint("/path/to/whole_image_model")
model_2 = EfficientLightning.load_from_checkpoint("/path/to/cropped_image_model")
model_3 = EfficientLightning.load_from_checkpoint("/path/to/negative_threshold_model")

for model in [model_1, model_2, model_3]:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

model = MultimodalModelLightning(
    m1=model_1,
    m2=model_2,
    m3=model_3,
    multimodal_model=MajorityVotingMultimodalModel,
    num_classes=1,
)

tumor_datamodule = BasicDataModule(
    dataset_class=MultimodalTumorDataset,
    dataset_init_kwargs=dict(
        datasets_paths=[
            "/path/to/usg_images",
            "/path/to/cropped_usg_images",
            "/path/to/negative_thresholded_usg_images",
        ],
        data_transforms=transformations,
        folders=["BUSI"],
    ),
    batch_size=7,
    num_workers=4,
    kfold_splits=5,
    n_split_kfold=0,
)

trainer = Trainer()

tumor_datamodule.setup()
val_loader = tumor_datamodule.val_dataloader()
trainer.validate(model, dataloaders=val_loader)
