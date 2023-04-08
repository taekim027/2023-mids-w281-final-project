import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import imageio.v3 as iio
import numpy as np
from itertools import chain

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

        return ImageDataset(groups.values())

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
