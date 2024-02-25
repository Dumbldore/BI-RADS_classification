import random
from typing import Optional, Tuple, Type

import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset, Subset

from data.datamodule_base import DataModuleBase


class BasicDataModule(DataModuleBase):
    def __init__(
        self,
        dataset_class: Type[Dataset],
        train_val_test_fractions: Tuple[float, float, float],
        batch_size: int = 1,
        num_workers: int = 0,
        split_seed: int = 42,
        shuffle: bool = True,
        kfold_splits: int = None,
        n_split_kfold: int = None,
        dataset_init_args: Optional[list] = None,
        dataset_init_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            dataset_class=dataset_class,
            dataset_init_args=dataset_init_args,
            dataset_init_kwargs=dataset_init_kwargs,
        )
        self.kfold_splits = kfold_splits
        self.n_split_kfold = n_split_kfold
        self._sets_fractions: Tuple[float, float, float] = train_val_test_fractions
        self._train_dataset: Subset = None
        self._val_dataset: Subset = None
        self._test_dataset: Subset = None
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers
        self._split_seed: int = split_seed
        self._shuffle: bool = shuffle

    def _validate_setup(self) -> None:
        super()._validate_setup()
        if not (0 <= sum(self._sets_fractions) <= 1):  # type: ignore
            raise ValueError(
                f"Sum of train, val and test fractions ({self._sets_fractions}) should"
                f" be from range [0, 1] but it is equal to "
                f"{sum(self._sets_fractions)}"
            )

    def _setup_subsets(self, dataset: Dataset) -> None:
        if self.kfold_splits is not None and self.n_split_kfold is not None:
            self._test_dataset = None
            indices = [i for i in range(len(dataset))]
            random.shuffle(indices)
            val_indices = indices[
                int(1 / self.kfold_splits * self.n_split_kfold * len(dataset)) : int(
                    1 / self.kfold_splits * (self.n_split_kfold + 1) * len(dataset)
                )
            ]
            train_indices = [i for i in range(len(dataset)) if i not in val_indices]
            self._val_dataset = torch.utils.data.Subset(dataset, val_indices)
            self._train_dataset = torch.utils.data.Subset(dataset, train_indices)
        else:
            test_size = int(len(dataset) * self._sets_fractions[2])
            val_size = int(len(dataset) * self._sets_fractions[1])
            train_size = (
                len(dataset) - val_size - test_size
                if sum(self._sets_fractions) == 1
                else int(len(dataset) * self._sets_fractions[0])
            )
            (
                self._train_dataset,
                self._val_dataset,
                self._test_dataset,
            ) = torch.utils.data.random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self._split_seed),
            )

    def train_dataloader(self) -> Optional[EVAL_DATALOADERS]:
        return (
            DataLoader(
                dataset=self._train_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                # sampler=sampler,
                pin_memory=True,
                # shuffle=False,
                shuffle=self._shuffle,
            )
            if len(self._train_dataset)
            else None
        )

    def val_dataloader(self) -> Optional[EVAL_DATALOADERS]:
        return (
            DataLoader(
                dataset=self._val_dataset,
                batch_size=self._batch_size,
                num_workers=1,
                # num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )
            if len(self._val_dataset)
            else None
        )

    def test_dataloader(self) -> Optional[EVAL_DATALOADERS]:
        return (
            DataLoader(
                dataset=self._test_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )
            if len(self._test_dataset)
            else None
        )

    def predict_dataloader(self) -> Optional[EVAL_DATALOADERS]:
        return (
            DataLoader(
                dataset=self._test_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                pin_memory=True,
                shuffle=False,
            )
            if len(self._test_dataset)
            else None
        )
