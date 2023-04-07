import os
from pathlib import Path
from typing import Iterable


class ImageAnnotation:
    def __init__(self, filepath: str) -> None:
        self.label = Path(filepath).parent.name
        self.filepath = filepath

    def load_image(self):
        pass


def load_images(directory: str) -> Iterable[ImageAnnotation]:
    assert os.path.isdir(directory), f'{directory} is not a valid directory'

    for filepath in Path(directory).rglob('*.png'):
        yield ImageAnnotation(filepath)