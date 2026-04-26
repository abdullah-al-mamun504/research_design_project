
# Dataset 3: GENERATE centered noisy DATASET

import os
import random
import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE      = 28
TRAIN_PER_CLASS = 2000
TEST_PER_CLASS  = 400
SHAPES   = ["circle", "square", "triangle", "hline", "vline"]

BASE_DIR = r"D:\dataset\simple_shapes_centered_noisy"  ## change directory  as required


def draw_shape(draw, shape, cx, cy, size):
    r = size // 2

    if shape == "circle":
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=0)

    elif shape == "square":
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=0)

    elif shape == "triangle":
        draw.polygon([(cx, cy-r), (cx-r, cy+r), (cx+r, cy+r)], fill=0)

    elif shape == "hline":
        draw.line([cx-r, cy, cx+r, cy], fill=0, width=4)

    elif shape == "vline":
        draw.line([cx, cy-r, cx, cy+r], fill=0, width=4)


# noise added for experiment

def add_noise(img, intensity=20):
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-intensity, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)



# doted noise added for experiment

def add_dots(draw, count=60):
    for _ in range(count):
        x = random.randint(0, IMAGE_SIZE - 1)
        y = random.randint(0, IMAGE_SIZE - 1)
        color = random.randint(100, 200)
        draw.ellipse([x, y, x+1, y+1], fill=color)


def create_image(shape):

    img  = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    draw = ImageDraw.Draw(img)
    size = random.randint(10, 18)
    #  centered
    cx = IMAGE_SIZE // 2
    cy = IMAGE_SIZE // 2

    draw_shape(draw, shape, cx, cy, size)

    # added noise using the code below for the experiment 
    add_dots(draw)
    img = add_noise(img, intensity=25)

    return img



def generate():

    for split, count in [("train", TRAIN_PER_CLASS), ("test", TEST_PER_CLASS)]:

        for shape in SHAPES:
            d = os.path.join(BASE_DIR, split, shape)
            os.makedirs(d, exist_ok=True)

            for i in range(count):
                img = create_image(shape)
                img.save(os.path.join(d, f"{shape}_{i}.png"))

    print(" Noisy centered dataset generated!")

generate()
