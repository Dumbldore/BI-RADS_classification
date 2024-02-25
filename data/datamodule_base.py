from abc import ABC, abstractmethod
from typing import Optional, Type

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset


class DataModuleBase(LightningDataModule, ABC):
    def __init__(
        self,
        dataset_class: Type[Dataset],
        dataset_init_args: Optional[list] = None,
        dataset_init_kwargs: Optional[dict] = None,
    ):
        super(DataModuleBase, self).__init__()
        self._dataset_class: Type[Dataset] = dataset_class
        self._dataset_init_args = dataset_init_args if dataset_init_args else []
        self._dataset_init_kwargs = dataset_init_kwargs if dataset_init_kwargs else {}

    def setup(self, stage: Optional[str] = None) -> None:
        self._validate_setup()
        dataset: Dataset = self._setup_dataset()
        self._setup_subsets(dataset)

    def _validate_setup(self) -> None:
        if not issubclass(self._dataset_class, Dataset):
            raise TypeError(
                f"'dataset_class' has to be a subclass of {Dataset.__name__}."
            )

    def _setup_dataset(self) -> Dataset:
        return self._dataset_class(
            *self._dataset_init_args, **self._dataset_init_kwargs
        )

    @abstractmethod
    def _setup_subsets(self, dataset: Dataset):
        raise NotImplementedError()
