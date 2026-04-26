
# Dataset 1: GENERATE off-centered image DATASET


import os
import random
from PIL import Image, ImageDraw

IMAGE_SIZE = 28
TRAIN_PER_CLASS = 2000
TEST_PER_CLASS = 400

SHAPES = ["circle", "square", "triangle", "hline", "vline"]

BASE_DIR = r"D:\dataset\simple_shapes"

def draw_shape(draw, shape, cx, cy, size):
    r = size // 2
    if shape == "circle":
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill="black")
    elif shape == "square":
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill="black")
    elif shape == "triangle":
        draw.polygon([(cx,cy-r),(cx-r,cy+r),(cx+r,cy+r)], fill="black")
    elif shape == "hline":
        draw.line([cx-r,cy,cx+r,cy], fill="black", width=2)
    elif shape == "vline":
        draw.line([cx,cy-r,cx,cy+r], fill="black", width=2)

def create_image(shape):
    img = Image.new("L", (28,28), "white")
    draw = ImageDraw.Draw(img)

    size = random.randint(10,18)
    m = size//2 + 2
    cx = random.randint(m,28-m)
    cy = random.randint(m,28-m)

    draw_shape(draw, shape, cx, cy, size)
    return img

def generate():
    for split, count in [("train", TRAIN_PER_CLASS), ("test", TEST_PER_CLASS)]:
        for shape in SHAPES:
            d = os.path.join(BASE_DIR, split, shape)
            os.makedirs(d, exist_ok=True)

            for i in range(count):
                img = create_image(shape)
                img.save(os.path.join(d, f"{i}.png"))

    print("Dataset ready!")

 generate() 

# ==============================
