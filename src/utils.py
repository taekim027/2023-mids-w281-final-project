from collections import defaultdict
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import imageio.v3 as iio
import numpy as np
from itertools import chain
import cv2
from random import shuffle

class ImageData:
    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        self.label = self.filepath.parent.name
        self.augmentation = self.filepath.parent.parent.name
        # Represents unique id of photo across augmentations
        self.id = self.filepath.name.split('.')[0].split('-')[0]
        self.is_sketch = self.filepath.parent.parent.parent.name == 'sketch'

    def load(self) -> np.ndarray:
        return iio.imread(self.filepath)

@dataclass
class ImageGroup:
    id: str
    label: str
    photos: List[ImageData] = field(default_factory=list)
    sketches: List[ImageData] = field(default_factory=list)

    def add(self, image: ImageData):
        group = self.sketches if image.is_sketch else self.photos
        group.append(image)

class ImageDataset(List[ImageGroup]):

    @staticmethod
    def from_directory(root_dir: str) -> 'ImageDataset':
        assert os.path.isdir(root_dir), f'{root_dir} is not a valid directory'

        print('Loading dataset...')
        groups: Dict[str, ImageGroup] = dict()
        for filepath in chain(Path(root_dir).rglob('*.jpg'), Path(root_dir).rglob('*.png')):

            image = ImageData(filepath)
            if image.id not in groups:
                groups[image.id] = ImageGroup(image.id, image.label)
            groups[image.id].add(image)

        print(f'Loaded {len(groups)} unique image IDs')
        return ImageDataset(groups.values())

    @property
    def num_classes(self) -> int:
        return len({l.label for l in self})

    @property
    def image_size(self) -> Tuple:
        return self[0].sketches[0].load().shape

    def filter(self, 
        photo_augmentations: List[str] = None, 
        sketch_augmentation: List[str] = None) -> 'ImageDataset':

        filtered = ImageDataset()
        for g in self:
            filtered_g = ImageGroup(g.id, g.label)
            filtered_g.photos = [p for p in g.photos if p.augmentation in photo_augmentations] \
                if photo_augmentations else g.photos
            filtered_g.sketches = [p for p in g.sketches if p.augmentation in sketch_augmentation] \
                if sketch_augmentation else g.sketches

            filtered.append(filtered_g)
        return filtered

    def split(self, ratios: List[float], preserve_labels: bool = True) -> List['ImageDataset']:

        def do_split(dataset: 'ImageDataset') -> List['ImageDataset']:
            shuffle(dataset)
            start = 0
            splits = []
            for ratio in ratios:
                end = start + int(len(dataset) * ratio)
                splits.append(ImageDataset(dataset[start:end]))
                start = end

            return splits

        if not preserve_labels:
            return do_split(self)

        # Group by label
        by_label: Dict[str, List[ImageGroup]] = defaultdict(list)
        for g in self:
            by_label[g.label].append(g)

        # Split by each label
        first_label, first_group = by_label.popitem()
        splits = do_split(first_group)
        for label, groups in by_label.items():
            for existing, additional in zip(splits, do_split(groups)):
                existing.extend(additional)

        return splits

    def get_labeled_sketch_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        Y = []
        
        for img in self:
            for i in img.sketches:
                fp = str(i.filepath.resolve())
                img = cv2.imread(fp)
                X.append(img)
                Y.append(i.label)
        print(f'Loaded {len(Y)} labeled sketches')
        return np.array(X), np.array(Y)
