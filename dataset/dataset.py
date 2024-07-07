import os
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset

from PIL import Image


class ICVLPDataset(Dataset):
    corpus_dict = {
        '<BLANK>': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
        'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
        'Y': 25, 'Z': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35, '0': 36
    }

    labels_dict = {v: k for k, v in corpus_dict.items()}

    def __init__(self,
                 root: str = 'data',
                 subset: str = 'train',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 size: Tuple[int, int] = (94, 24),
                 corpus_dict: dict[str: int] = None
                 ) -> None:
        assert subset in ['train', 'test', 'val'], f'Subset must be "train", "test", or "val". Got "{subset}"'

        super().__init__()
        self.root = root
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform
        self.size = size

        if corpus_dict is not None:
            self.corpus_dict = corpus_dict
            self.labels_dict = {v: k for k, v in corpus_dict.items()}

        # if download:
        #     self.download()
        #
        # if not self._check_exists():
        #     raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def load_data(self):
        images_path = os.path.join(self.root, 'icvlp', self.subset)
        images = []
        labels = []
        for filename in os.listdir(images_path):
            # Images
            img_path = os.path.join(images_path, filename)
            img = Image.open(img_path)
            img = img.resize(self.size)
            images.append(img)

            # Labels
            label = filename.split('_')[0]
            labels.append(label)

        return images, labels

    def download(self):
        raise NotImplementedError  # TODO: Download dataset if not exists in filesystem

    def _check_exists(self):
        raise NotImplementedError  # TODO: Check if dataset already downloaded
