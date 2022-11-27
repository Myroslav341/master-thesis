import os
import random
import shutil
from dataclasses import dataclass
from typing import List

from bs4 import BeautifulSoup

config_file = "../data/annotations/annotations/trainval.txt"

new_config_file = "train_data.txt"


@dataclass
class BoundaryBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __str__(self):
        return f"[{self.x_min}; {self.y_min}; {self.x_max}; {self.y_max}]"

    def __repr__(self):
        return self.__str__()


@dataclass
class TrainImage:
    image_path: str
    boundary_box: BoundaryBox
    pet_class: int


def read_xml(path: str) -> BoundaryBox:
    with open(path, 'r') as f:
        data = f.read()

    xml_data = BeautifulSoup(data, "xml")

    boundary_box = xml_data.find("bndbox")

    x_min = int(boundary_box.find("xmin").contents[0])
    y_min = int(boundary_box.find("ymin").contents[0])
    x_max = int(boundary_box.find("xmax").contents[0])
    y_max = int(boundary_box.find("ymax").contents[0])

    return BoundaryBox(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )


def create_train_file():
    with open(config_file) as f:
        data = f.read()

    train_images: List[TrainImage] = []

    for train_image in data.splitlines():
        image_name = train_image.split()[0]

        pet_class = 0 if int(train_image.split()[2]) == 1 else 1

        annotation_path = f"data/annotations/annotations/xmls/{image_name}.xml"
        image_path = f"data/images/images/{image_name}.jpg"

        if not os.path.exists(annotation_path):
            print(f"annotation path doesn't exist for image {image_name}")
            continue

        if not os.path.exists(image_path):
            print(f"image path doesn't exist for image {image_path}")
            continue

        box = read_xml(annotation_path)

        train_images.append(
            TrainImage(
                image_path=image_path,
                boundary_box=box,
                pet_class=pet_class
            )
        )

    random.shuffle(train_images)
    with open(new_config_file, "w") as f:
        for image in train_images:
            if image.pet_class == 1:
                f.write(f"{image.image_path},{image.pet_class},{image.boundary_box}\n")


def copy_train_images():
    f = open(new_config_file, "r")

    train_images = f.readlines()
    new_lies = []

    f.close()

    for train_image in train_images:
        image_path = train_image.split(",")[0]
        image_name = image_path.split("/")[-1]

        shutil.copyfile(image_path, f"data/train/{image_name}")

        new_lies.append(train_image.replace("data/images/images/", "data/train/"))

    f = open(new_config_file, "w")

    f.writelines(new_lies)

    f.close()


if __name__ == '__main__':
    copy_train_images()
