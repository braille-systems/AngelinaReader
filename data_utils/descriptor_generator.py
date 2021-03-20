from typing import Sequence
from enum import Enum
from pathlib import Path
from os import path


class ImgDescriptorType(Enum):
    train = "train"
    val = "val"


def add_images_descriptor(directory: Path, descriptor_type: ImgDescriptorType,
                          img_patterns: Sequence[str] = ("*.labeled.jpg",)) -> None:
    """
    For given patterns and directory, write all filenames matching any of those patterns in that directory or its
    subdirectories into a file "val.txt" or "train.txt" (depends on `descriptor_type`)
    :param directory: a directory to search files (recursively) and to store "val.txt"/"train.txt"
    :param descriptor_type: defines the output filename: "val.txt" or "train.txt"
    :param img_patterns: a list of patterns, e. g. ["*.jpg", "img*.png"] (case insensitive)
    :return: None
    """
    img_filenames = [path.basename(str(img_filename))
                     for pattern in img_patterns
                     for img_filename in Path.rglob(directory, pattern)]  # patterns are case insensitive
    with open(str(directory / (descriptor_type.value + ".txt")), "w") as descriptor_file:
        descriptor_file.writelines([filename + "\n" for filename in img_filenames])


if __name__ == "__main__":
    for unlabeled_sub_dir in ("golubina", "plates"):
        directory = (Path("../brl_ocr/data/unlabeled") / unlabeled_sub_dir) / "with_pseudolabels"
        add_images_descriptor(directory, ImgDescriptorType.train)
