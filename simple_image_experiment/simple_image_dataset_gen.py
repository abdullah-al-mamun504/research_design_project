import os
import random
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
IMAGE_SIZE = 64              # 64x64 pixels
NUM_IMAGES_PER_SHAPE = 100   # increase for better training
SHAPES = ["circle", "square", "triangle", "hline", "vline"] # 5 classes shapes

OUTPUT_DIR = r"D:\dataset\low_reg_Dataset_64by64"   # Change this to your desired output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DRAW SHAPES
# ─────────────────────────────────────────────
def draw_shape(draw, shape, cx, cy, size):
    r = size // 2

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="black")

    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill="black")

    elif shape == "triangle":
        points = [
            (cx, cy - r),
            (cx - r, cy + r),
            (cx + r, cy + r),
        ]
        draw.polygon(points, fill="black")

    elif shape == "hline":
        width = max(3, size // 8)
        draw.line([cx - r, cy, cx + r, cy], fill="black", width=width)

    elif shape == "vline":
        width = max(3, size // 8)
        draw.line([cx, cy - r, cx, cy + r], fill="black", width=width)


# ─────────────────────────────────────────────
# CREATE SINGLE IMAGE
# ─────────────────────────────────────────────
def create_image(shape):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white") # white background
    draw = ImageDraw.Draw(image)
    #color = random.choice(["black"])  # extend list for color variety, e.g. ["black", "red", "blue"]
    # Better size range for visibility
    size = random.randint(35, 55)  # increase size for better visibility in 64x64 images

    # Ensure shape stays inside image
    margin = size // 2 + 2          # add a small margin to prevent clipping
    cx = random.randint(margin, IMAGE_SIZE - margin)
    cy = random.randint(margin, IMAGE_SIZE - margin)

    draw_shape(draw, shape, cx, cy, size)

    return image


# ─────────────────────────────────────────────
# GENERATE DATASET
# ─────────────────────────────────────────────
def generate_dataset():
    print("Generating dataset...")

    for shape in SHAPES:
        shape_dir = os.path.join(OUTPUT_DIR, shape)
        os.makedirs(shape_dir, exist_ok=True)

        for i in range(NUM_IMAGES_PER_SHAPE):
            img = create_image(shape)
            img.save(os.path.join(shape_dir, f"{shape}_{i}.png"))

        print(f"✔ {shape} done")

    print("\n✅ Dataset generation complete!")
    print(f"Saved at: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    generate_dataset()