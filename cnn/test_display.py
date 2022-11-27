from dataclasses import dataclass

from PIL import Image, ImageDraw
from bs4 import BeautifulSoup


IMAGE_ID = "101"
BASE_PATH = "/"
IMAGE_NAME = f"Abyssinian_{IMAGE_ID}"
IMAGE_PATH = f"{BASE_PATH}/data/images/images/{IMAGE_NAME}.jpg"
ANNOTATION_PATH = f"{BASE_PATH}/data/annotations/annotations/xmls/{IMAGE_NAME}.xml"


@dataclass
class BoundaryBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


def show_image():
    im = Image.open(IMAGE_PATH)

    im.show()


def show_image_with_boundary_box(boundary: BoundaryBox):
    im = Image.open(IMAGE_PATH)

    img1 = ImageDraw.Draw(im)
    img1.rectangle((boundary.x_min, boundary.y_min, boundary.x_max, boundary.y_max), outline="red", width=10)

    im.show()


def read_xml() -> BoundaryBox:
    with open(ANNOTATION_PATH, 'r') as f:
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


if __name__ == '__main__':
    boundary_box = read_xml()
    show_image_with_boundary_box(boundary_box)
