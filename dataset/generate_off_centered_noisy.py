
# Dataset 4: GENERATE off centered noisy DATASET



import os
import random
import numpy as np
from PIL import Image, ImageDraw


IMAGE_SIZE      = 28
TRAIN_PER_CLASS = 2000
TEST_PER_CLASS  = 400
SHAPES          = ["circle", "square", "triangle", "hline", "vline"]
BASE_DIR        = r"D:\dataset\simple_shapes_off_centered_noisy"


def add_noise(img, intensity=25):
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-intensity, intensity, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_dots(draw, count=60):
    for _ in range(count):
        x = random.randint(0, IMAGE_SIZE - 1)
        y = random.randint(0, IMAGE_SIZE - 1)
        color = random.randint(100, 200)
        draw.ellipse([x, y, x+1, y+1], fill=color)

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


def create_image(shape):
    # Create blank white image
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    draw = ImageDraw.Draw(img)

    # Determine random size and random OFF-CENTERED position
    size = random.randint(10, 16)
    margin = size // 2 + 2
    cx = random.randint(margin, IMAGE_SIZE - margin)
    cy = random.randint(margin, IMAGE_SIZE - margin)

    draw_shape(draw, shape, cx, cy, size)

    #  create Noise Dots first, then Intensity or Grain
    add_dots(draw, count=50)
    img = add_noise(img, intensity=25)
  
    return img

def generate():
    print(f"Generating dataset at: {BASE_DIR}")
    for split, count in [("train", TRAIN_PER_CLASS), ("test", TEST_PER_CLASS)]:
        for shape in SHAPES:
            d = os.path.join(BASE_DIR, split, shape)
            os.makedirs(d, exist_ok=True)

            for i in range(count):
                img = create_image(shape)
                img.save(os.path.join(d, f"{shape}_{i}.png"))
              
    print("Off-centered noisy dataset generated successfully!")

if __name__ == "__main__":
    generate()
