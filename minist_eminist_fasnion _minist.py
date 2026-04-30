import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================================================
#  COMMON TRANSFORM  (same for all three datasets)
#  28x28 grayscale → tensor → normalized to [-1, 1]
# ============================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ============================================================
#  1. ORIGINAL MNIST  (for reference / comparison)
# ============================================================
mnist_path = r"D:\Research design course\minist_dataset_experiment"

mnist_train = datasets.MNIST(root=mnist_path, train=True,  download=True, transform=transform)
mnist_test  = datasets.MNIST(root=mnist_path, train=False, download=True, transform=transform)

mnist_train_loader = DataLoader(mnist_train, batch_size=256, shuffle=True)
mnist_test_loader  = DataLoader(mnist_test,  batch_size=256, shuffle=False)

print("===== MNIST =====")
print(f"Train: {len(mnist_train)} | Test: {len(mnist_test)}")
print(f"Classes: {mnist_train.classes}")


# ============================================================
#  2. FASHION-MNIST
#     Same structure as MNIST — just swap datasets.MNIST
#     with datasets.FashionMNIST
#     10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
#                 Sandal, Shirt, Sneaker, Bag, Ankle boot
# ============================================================
fashion_path = r"D:\Research design course\fashion_mnist_dataset"

fashion_train = datasets.FashionMNIST(root=fashion_path, train=True,  download=True, transform=transform)
fashion_test  = datasets.FashionMNIST(root=fashion_path, train=False, download=True, transform=transform)

fashion_train_loader = DataLoader(fashion_train, batch_size=256, shuffle=True)
fashion_test_loader  = DataLoader(fashion_test,  batch_size=256, shuffle=False)

print("\n===== Fashion-MNIST =====")
print(f"Train: {len(fashion_train)} | Test: {len(fashion_test)}")
print(f"Classes: {fashion_train.classes}")


# ============================================================
#  3. EMNIST
#     Extra parameter: split=
#     Options: 'byclass' | 'bymerge' | 'balanced' |
#              'letters' | 'digits'  | 'mnist'
#
#     Most commonly used splits:
#       'balanced' → 47 classes, digits + letters (recommended)
#       'letters'  → 26 classes, A-Z only
#       'digits'   → 10 classes, 0-9 (like MNIST but larger)
#       'byclass'  → 62 classes (all digits + upper + lower)
#
#  ⚠️ NOTE: EMNIST images are rotated 90° and flipped.
#     The extra transforms below fix the orientation.
# ============================================================
emnist_path = r"D:\Research design course\emnist_dataset"

emnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    # Fix EMNIST orientation (rotate + flip to human-readable)
    transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=[1, 2])),
    transforms.Lambda(lambda x: torch.flip(x, dims=[2]))
])

# --- Split: 'balanced' (recommended starting point) ---
emnist_train = datasets.EMNIST(root=emnist_path, split='balanced', train=True,  download=True, transform=emnist_transform)
emnist_test  = datasets.EMNIST(root=emnist_path, split='balanced', train=False, download=True, transform=emnist_transform)

emnist_train_loader = DataLoader(emnist_train, batch_size=256, shuffle=True)
emnist_test_loader  = DataLoader(emnist_test,  batch_size=256, shuffle=False)

print("\n===== EMNIST (balanced split) =====")
print(f"Train: {len(emnist_train)} | Test: {len(emnist_test)}")
print(f"Number of classes: {len(emnist_train.classes)}")
print(f"Classes: {emnist_train.classes}")

# --- Switching to a different split is just one word change ---
# emnist_letters_train = datasets.EMNIST(root=emnist_path, split='letters',  train=True, download=True, transform=emnist_transform)
# emnist_digits_train  = datasets.EMNIST(root=emnist_path, split='digits',   train=True, download=True, transform=emnist_transform)
# emnist_byclass_train = datasets.EMNIST(root=emnist_path, split='byclass',  train=True, download=True, transform=emnist_transform)


# ============================================================
#  QUICK SANITY CHECK — verify batch shapes for all 3
# ============================================================
print("\n===== Batch Shape Sanity Check =====")

imgs, lbls = next(iter(mnist_train_loader))
print(f"MNIST        — images: {imgs.shape} | labels: {lbls.shape}")

imgs, lbls = next(iter(fashion_train_loader))
print(f"Fashion-MNIST— images: {imgs.shape} | labels: {lbls.shape}")

imgs, lbls = next(iter(emnist_train_loader))
print(f"EMNIST       — images: {imgs.shape} | labels: {lbls.shape}")
