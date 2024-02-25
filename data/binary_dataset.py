import os
from typing import Callable, Optional

from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import find_classes


class TumorDataset(ImageFolder):
    """
    A subclass of torchvision's ImageFolder dataset for handling tumor images.
    This class is designed to work with a directory structure where images are grouped into subdirectories.
    """

    def __init__(
        self,
        folders,
        dataset_path,
        data_transforms,
    ) -> None:
        """
        Initialize the TumorDataset.

        :param folders: A list of folder names that contain the images.
        :param dataset_path: The path to the root directory of the dataset.
        :param data_transforms: A list of torchvision transforms to be applied to the images.
        """
        self.subsets_folders = folders
        super(TumorDataset, self).__init__(
            root=dataset_path,
            transform=(
                transforms.Compose(data_transforms)
                if data_transforms is not None
                else None
            ),
        )

    def find_classes(self, directory):
        """
        Find all unique classes in the dataset.

        :param directory: The directory to search for classes.
        :return: A tuple containing a list of class names and a dictionary mapping class names to indices.
        """
        found_subsets_classes = [
            find_classes(os.path.join(directory, subset_dir))
            for subset_dir in self.subsets_folders
        ]
        all_classes_names = list(
            {
                class_name
                for subset_classes in found_subsets_classes
                for class_name in subset_classes[0]
            }
        )
        all_classes_names_to_idx = {
            k: v
            for subset_classes in found_subsets_classes
            for (k, v) in subset_classes[1].items()
        }
        return all_classes_names, all_classes_names_to_idx

    def make_dataset(
        self,
        directory,
        class_to_idx,
        extensions: Optional[tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """
        Create a dataset of (sample, target) tuples.

        :param directory: The directory containing the images.
        :param class_to_idx: A dictionary mapping class names to indices.
        :param extensions: A tuple of valid file extensions for the images.
        :param is_valid_file: A function that takes a file path and returns True if the file is valid, False otherwise.
        :return: A list of (sample, target) tuples.
        """
        return [
            instance
            for subset_dir in self.subsets_folders
            for instance in super(TumorDataset, self).make_dataset(
                os.path.join(directory, subset_dir),
                class_to_idx,
                extensions,
                is_valid_file,
            )
        ]


class MultimodalTumorDataset(TumorDataset):
    """
    A subclass of TumorDataset for handling multimodal tumor images.
    This class is designed to work with multiple datasets, each representing a different modality.
    """

    def __init__(
        self,
        folders,
        datasets_paths,
        data_transforms,
    ) -> None:
        """
        Initialize the MultimodalTumorDataset.

        :param folders: A list of folder names that contain the images.
        :param datasets_paths: A list of paths to the root directories of the datasets.
        :param data_transforms: A list of torchvision transforms to be applied to the images.
        """
        self.paths = datasets_paths
        self.data_transforms = data_transforms
        super().__init__(
            dataset_path=datasets_paths[0],
            data_transforms=data_transforms,
            folders=folders,
        )

    def __getitem__(self, index: int):
        """
        Get an item from the dataset.

        :param index: The index of the item.
        :return: A tuple containing the transformed image(s) and the target.
        """
        path, target = self.samples[index]
        X = [
            self.data_transforms(
                Image.open(path.replace(self.paths[0], dataset)).convert("RGB")
            )
            for dataset in self.paths
        ]
        return (X[0] if len(X) == 1 else X), target
