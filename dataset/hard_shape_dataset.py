# =====================================================
# HARD SHAPE DATASET GENERATOR
# Generates 28x28 images with:
#   - 8 shape classes
#   - 7 foreground colors + black/white mode
#   - Random background colors
#   - Gaussian noise, random dots, zigzag distractors
#   - Random position, size, rotation
#   - Filled and outline render styles
#   - Uneven/irregular geometry variants
# =====================================================


### not yet implemented ## this is just idea

import os
import random
import numpy as np
from PIL import Image, ImageDraw
import math

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
OUTPUT_ROOT = r"D:\dataset\hard_shapes"
IMG_SIZE    = 28
TRAIN_COUNT = 10000
TEST_COUNT  = 2000

CLASSES = [
    "circle",
    "square",
    "triangle",
    "hline",
    "vline",
    "pentagon",
    "ellipse",
    "cross",
]

#  foreground colors + black/white mode
COLORS = [
    (220,  50,  50),   # red
    (50,  130, 220),   # blue
    ( 40, 180,  80),   # green
    (230, 160,  20),   # amber
    (160,  50, 210),   # purple
    ( 20, 180, 180),   # teal
    (220,  90, 160),   # pink
]

NOISE_LEVEL     = 0.35   # probability of adding noise layer
DOT_LEVEL       = 0.35   # probability of adding random dots
ZIGZAG_LEVEL    = 0.30   # probability of adding zigzag distractor
BLOB_LEVEL      = 0.25   # probability of random blob
OUTLINE_PROB    = 0.40   # probability shape is outline (not filled)
BW_PROB         = 0.20   # probability image is black/white


# ──────────────────────────────────────────
# COLOR HELPERS
# ──────────────────────────────────────────
def rand_color():
    return random.choice(COLORS)

def rand_bg():
    """Random background: dark, light, or mid-tone."""
    mode = random.random()
    if mode < 0.33:
        v = random.randint(10, 60)
        return (v, v, v)         # dark
    elif mode < 0.66:
        v = random.randint(180, 245)
        return (v, v, v)         # light
    else:
        return (                  # random color bg
            random.randint(60, 200),
            random.randint(60, 200),
            random.randint(60, 200),
        )

def ensure_contrast(fg, bg, min_dist=80):
    """Re-roll fg until it contrasts enough with bg."""
    for _ in range(20):
        dist = sum(abs(f - b) for f, b in zip(fg, bg))
        if dist >= min_dist:
            return fg
        fg = rand_color()
    return fg

def to_bw(img_arr):
    """Convert RGB array to grayscale-RGB (black/white render)."""
    gray = np.mean(img_arr, axis=2, keepdims=True).astype(np.uint8)
    return np.concatenate([gray, gray, gray], axis=2)


# ──────────────────────────────────────────
# DISTRACTOR GENERATORS
# ──────────────────────────────────────────
def add_gaussian_noise(arr, sigma=18):
    noise = np.random.normal(0, sigma, arr.shape).astype(np.int16)
    return np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def add_random_dots(draw, n_dots=None, color=(128, 128, 128)):
    if n_dots is None:
        n_dots = random.randint(4, 20)
    for _ in range(n_dots):
        x = random.randint(0, IMG_SIZE - 1)
        y = random.randint(0, IMG_SIZE - 1)
        r = random.randint(0, 1)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

def add_zigzag(draw, color):
    n_points = random.randint(4, 8)
    x = random.randint(0, IMG_SIZE // 2)
    pts = []
    for i in range(n_points):
        x += random.randint(2, 6)
        y  = random.randint(2, IMG_SIZE - 2) if i % 2 == 0 else random.randint(2, IMG_SIZE - 2)
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=1)

def add_blob(draw, color):
    cx = random.randint(4, IMG_SIZE - 4)
    cy = random.randint(4, IMG_SIZE - 4)
    rx = random.randint(2, 6)
    ry = random.randint(2, 6)
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=color)


# ──────────────────────────────────────────
# SHAPE DRAWERS
# ──────────────────────────────────────────
def random_pos_size(min_sz=6, max_sz=16):
    """Return (cx, cy, half_size) with shape fully inside image."""
    sz = random.randint(min_sz, max_sz)
    margin = sz + 1
    cx = random.randint(margin, IMG_SIZE - margin)
    cy = random.randint(margin, IMG_SIZE - margin)
    return cx, cy, sz

def rotate_points(pts, cx, cy, angle_deg):
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    out = []
    for x, y in pts:
        dx, dy = x - cx, y - cy
        out.append((cx + dx*cos_a - dy*sin_a,
                    cy + dx*sin_a + dy*cos_a))
    return out

def draw_circle(draw, fg, outline_only):
    cx, cy, sz = random_pos_size()
    box = [cx-sz, cy-sz, cx+sz, cy+sz]
    if outline_only:
        draw.ellipse(box, outline=fg, width=random.randint(1,2))
    else:
        draw.ellipse(box, fill=fg)

def draw_ellipse(draw, fg, outline_only):
    cx, cy, sz = random_pos_size()
    rx = sz
    ry = random.randint(max(2, sz//3), sz - 1)
    box = [cx-rx, cy-ry, cx+rx, cy+ry]
    if outline_only:
        draw.ellipse(box, outline=fg, width=random.randint(1,2))
    else:
        draw.ellipse(box, fill=fg)

def draw_square(draw, fg, outline_only):
    cx, cy, sz = random_pos_size()
    # optional slight rotation to make "uneven"
    angle = random.choice([0, 0, 0, 15, 30, 45])
    pts = [(-sz,-sz),(sz,-sz),(sz,sz),(-sz,sz)]
    pts = rotate_points([(cx+x, cy+y) for x,y in pts], cx, cy, angle)
    if outline_only:
        draw.polygon(pts, outline=fg)
    else:
        draw.polygon(pts, fill=fg)

def draw_triangle(draw, fg, outline_only):
    cx, cy, sz = random_pos_size()
    angle = random.randint(0, 359)
    base_pts = [(0, -sz), (-sz, sz), (sz, sz)]
    pts = rotate_points([(cx+x, cy+y) for x,y in base_pts], cx, cy, angle)
    if outline_only:
        draw.polygon(pts, outline=fg)
    else:
        draw.polygon(pts, fill=fg)

def draw_hline(draw, fg, outline_only):
    cy  = random.randint(6, IMG_SIZE - 6)
    x0  = random.randint(1, 6)
    x1  = random.randint(IMG_SIZE - 6, IMG_SIZE - 1)
    w   = random.randint(1, 3)
    # slight tilt for variety
    tilt = random.randint(-3, 3)
    draw.line([(x0, cy + tilt), (x1, cy - tilt)], fill=fg, width=w)

def draw_vline(draw, fg, outline_only):
    cx  = random.randint(6, IMG_SIZE - 6)
    y0  = random.randint(1, 6)
    y1  = random.randint(IMG_SIZE - 6, IMG_SIZE - 1)
    w   = random.randint(1, 3)
    tilt = random.randint(-3, 3)
    draw.line([(cx + tilt, y0), (cx - tilt, y1)], fill=fg, width=w)

def draw_pentagon(draw, fg, outline_only):
    cx, cy, sz = random_pos_size()
    angle = random.randint(0, 359)
    pts = []
    for i in range(5):
        a = math.radians(angle + i * 72)
        pts.append((cx + sz * math.cos(a), cy + sz * math.sin(a)))
    if outline_only:
        draw.polygon(pts, outline=fg)
    else:
        draw.polygon(pts, fill=fg)

def draw_cross(draw, fg, outline_only):
    cx, cy, sz = random_pos_size(min_sz=5)
    arm = max(2, sz // 3)
    # cross as two overlapping rectangles
    h_box = [cx-sz, cy-arm, cx+sz, cy+arm]
    v_box = [cx-arm, cy-sz, cx+arm, cy+sz]
    if outline_only:
        draw.rectangle(h_box, outline=fg)
        draw.rectangle(v_box, outline=fg)
    else:
        draw.rectangle(h_box, fill=fg)
        draw.rectangle(v_box, fill=fg)


SHAPE_DRAWERS = {
    "circle":   draw_circle,
    "ellipse":  draw_ellipse,
    "square":   draw_square,
    "triangle": draw_triangle,
    "hline":    draw_hline,
    "vline":    draw_vline,
    "pentagon": draw_pentagon,
    "cross":    draw_cross,
}


# ──────────────────────────────────────────
# SINGLE IMAGE GENERATOR
# ──────────────────────────────────────────
def generate_image(class_name):
    bg  = rand_bg()
    fg  = rand_color()
    fg  = ensure_contrast(fg, bg)
    bw_mode     = random.random() < BW_PROB
    outline_only = random.random() < OUTLINE_PROB

    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)

    # ── Distractors (drawn BEFORE main shape so shape is on top) ──
    dist_color = rand_color()
    dist_color = ensure_contrast(dist_color, bg, min_dist=60)

    if random.random() < BLOB_LEVEL:
        add_blob(draw, dist_color)
    if random.random() < DOT_LEVEL:
        add_random_dots(draw, color=dist_color)
    if random.random() < ZIGZAG_LEVEL:
        add_zigzag(draw, dist_color)

    # ── Main shape ──
    SHAPE_DRAWERS[class_name](draw, fg, outline_only)

    # ── Post-draw distractors (on top of shape, partial occlusion) ──
    if random.random() < DOT_LEVEL * 0.4:
        add_random_dots(draw, n_dots=random.randint(1, 5), color=dist_color)

    arr = np.array(img)

    # ── Gaussian pixel noise ──
    if random.random() < NOISE_LEVEL:
        arr = add_gaussian_noise(arr, sigma=random.randint(10, 30))

    # ── Black/white mode ──
    if bw_mode:
        arr = to_bw(arr)

    return Image.fromarray(arr)


# ──────────────────────────────────────────
# DATASET BUILDER
# ──────────────────────────────────────────
def build_split(split_name, total_count):
    per_class = total_count // len(CLASSES)
    split_dir = os.path.join(OUTPUT_ROOT, split_name)

    for cls in CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(per_class):
            img = generate_image(cls)
            img.save(os.path.join(cls_dir, f"{i:05d}.png"))

        print(f"  [{split_name}] {cls}: {per_class} images saved")

def main():
    print(f"Generating hard shape dataset → {OUTPUT_ROOT}")
    print(f"Classes: {CLASSES}")
    print(f"Train: {TRAIN_COUNT}  |  Test: {TEST_COUNT}")
    print(f"Noise: {NOISE_LEVEL:.0%}  Dots: {DOT_LEVEL:.0%}  "
          f"Zigzag: {ZIGZAG_LEVEL:.0%}  Outline: {OUTLINE_PROB:.0%}  BW: {BW_PROB:.0%}")
    print()

    print("Building train split...")
    build_split("train", TRAIN_COUNT)

    print("\nBuilding test split...")
    build_split("test", TEST_COUNT)

    print(f"\nDone. Dataset saved to: {OUTPUT_ROOT}")
    print("Load with:  datasets.ImageFolder(root=path + '\\\\train', transform=transform)")

if __name__ == "__main__":
    main()
