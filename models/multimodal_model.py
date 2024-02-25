import torch
import torch.nn as nn
from torch import Tensor


class SimpleMultimodalModel(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()


class AverageSimpleMultimodalModel(SimpleMultimodalModel):
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(
            torch.stack([model(x[i]) for i, model in enumerate(self.models)]), dim=0
        )


class MaxSimpleMultimodalModel(SimpleMultimodalModel):
    def forward(self, x: Tensor) -> Tensor:
        return torch.max(
            torch.stack([model(x[i]) for i, model in enumerate(self.models)]), dim=0
        )


class MajorityVotingMultimodalModel(SimpleMultimodalModel):
    def forward(self, x: Tensor) -> Tensor:
        outputs = [
            torch.where(torch.sigmoid(model(x[i])) > 0.5, 1, 0)
            for i, model in enumerate(self.models)
        ]
        return (torch.sum(torch.stack(outputs), dim=0) >= 2).float()
