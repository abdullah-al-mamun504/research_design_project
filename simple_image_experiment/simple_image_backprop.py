import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────
DATA_DIR   = r"D:\dataset\low_reg_Dataset_64by64"   # same as your generator
IMAGE_SIZE = 64
SHAPES     = ["circle", "square", "triangle", "hline", "vline"]
NUM_CLASSES = len(SHAPES)          # 5

# Network architecture
INPUT_SIZE  = IMAGE_SIZE * IMAGE_SIZE  # 4096 (flattened pixels)
HIDDEN_SIZE = 128
OUTPUT_SIZE = NUM_CLASSES

# Hyper-parameters
LEARNING_RATE = 0.01
EPOCHS        = 500
TEST_SPLIT    = 0.2   # 20 % for testing


# ─────────────────────────────────────────────
#  1. LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset():
    X, y = [], []
    for label, shape in enumerate(SHAPES):
        folder = os.path.join(DATA_DIR, shape)
        for fname in os.listdir(folder):
            if not fname.endswith(".png"):
                continue
            img = Image.open(os.path.join(folder, fname)).convert("L")  # grayscale
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0  # normalize 0–1
            X.append(pixels)
            y.append(label)
    return np.array(X), np.array(y)

# One-hot encode labels  e.g. label 2 → [0, 0, 1, 0, 0]
def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


# ─────────────────────────────────────────────
#  2. ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)          # hidden layer activation

def relu_deriv(z):
    return (z > 0).astype(float)     # derivative of ReLU

def softmax(z):
    # Subtract max for numerical stability
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)   # output probabilities


# ─────────────────────────────────────────────
#  3. INITIALIZE WEIGHTS  (He initialization)
# ─────────────────────────────────────────────
def init_weights():
    np.random.seed(42)
    W1 = np.random.randn(INPUT_SIZE,  HIDDEN_SIZE) * np.sqrt(2 / INPUT_SIZE)
    b1 = np.zeros((1, HIDDEN_SIZE))
    W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2 / HIDDEN_SIZE)
    b2 = np.zeros((1, OUTPUT_SIZE))
    return W1, b1, W2, b2


# ─────────────────────────────────────────────
#  4. FORWARD PASS
#     Input → Hidden (ReLU) → Output (Softmax)
# ─────────────────────────────────────────────
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1        # linear combination, shape (N, 128)
    A1 = relu(Z1)            # ReLU activation,    shape (N, 128)
    Z2 = A1 @ W2 + b2        # linear combination, shape (N, 5)
    A2 = softmax(Z2)         # probabilities,       shape (N, 5)
    return Z1, A1, Z2, A2


# ─────────────────────────────────────────────
#  5. LOSS  — Cross-Entropy
#     L = -mean( y_true * log(y_pred) )
# ─────────────────────────────────────────────
def cross_entropy_loss(y_pred, y_true_onehot):
    eps = 1e-9                          # avoid log(0)
    n   = y_true_onehot.shape[0]
    return -np.sum(y_true_onehot * np.log(y_pred + eps)) / n


# ─────────────────────────────────────────────
#  6. BACKWARD PASS  (Backpropagation)
#
#  Chain rule — gradients flow right → left:
#
#   dL/dW2 = A1ᵀ · dZ2
#   dL/dW1 = Xᵀ  · dZ1
#
#  where:
#   dZ2 = A2 - y_true          (softmax + cross-entropy gradient)
#   dZ1 = (dZ2 · W2ᵀ) * relu'(Z1)
# ─────────────────────────────────────────────
def backward(X, y_onehot, Z1, A1, A2, W1, b1, W2, b2):
    n = X.shape[0]

    # --- Output layer gradient ---
    dZ2 = A2 - y_onehot                   # (N, 5)
    dW2 = A1.T @ dZ2 / n                  # (128, 5)
    db2 = dZ2.mean(axis=0, keepdims=True) # (1, 5)

    # --- Hidden layer gradient ---
    dA1 = dZ2 @ W2.T                      # (N, 128)
    dZ1 = dA1 * relu_deriv(Z1)            # (N, 128)  ← chain rule through ReLU
    dW1 = X.T @ dZ1 / n                   # (4096, 128)
    db1 = dZ1.mean(axis=0, keepdims=True) # (1, 128)

    return dW1, db1, dW2, db2


# ─────────────────────────────────────────────
#  7. WEIGHT UPDATE  (Gradient Descent)
#     W = W - lr * dW
# ─────────────────────────────────────────────
def update_weights(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2


# ─────────────────────────────────────────────
#  8. ACCURACY HELPER
# ─────────────────────────────────────────────
def accuracy(X, y, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    preds = np.argmax(A2, axis=1)   # pick class with highest probability
    return np.mean(preds == y) * 100


# ─────────────────────────────────────────────
#  9. TRAINING LOOP
# ─────────────────────────────────────────────
def train():
    # -- Load & split --
    print("Loading dataset...")
    X, y = load_dataset()
    y_oh = one_hot(y, NUM_CLASSES)

    X_train, X_test, y_train, y_test, yoh_train, yoh_test = train_test_split(
        X, y, y_oh, test_size=TEST_SPLIT, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}  |  Test samples: {len(X_test)}\n")

    # -- Initialize --
    W1, b1, W2, b2 = init_weights()

    # -- Epoch loop --
    for epoch in range(1, EPOCHS + 1):

        # Forward
        Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

        # Loss
        loss = cross_entropy_loss(A2, yoh_train)

        # Backward
        dW1, db1, dW2, db2 = backward(X_train, yoh_train, Z1, A1, A2,
                                       W1, b1, W2, b2)

        # Update
        W1, b1, W2, b2 = update_weights(W1, b1, W2, b2,
                                         dW1, db1, dW2, db2,
                                         LEARNING_RATE)

        # Log every 10 epochs
        if epoch % 5 == 0:
            train_acc = accuracy(X_train, y_train, W1, b1, W2, b2)
            test_acc  = accuracy(X_test,  y_test,  W1, b1, W2, b2)
            print(f"Epoch {epoch:4d}/{EPOCHS} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    # -- Final report --
    print("\n── Final Results ──")
    print(f"Train Accuracy : {accuracy(X_train, y_train, W1, b1, W2, b2):.1f}%")
    print(f"Test  Accuracy : {accuracy(X_test,  y_test,  W1, b1, W2, b2):.1f}%")

    # -- Per-class accuracy --
    print("\n── Per-Class Test Accuracy ──")
    _, _, _, A2 = forward(X_test, W1, b1, W2, b2)
    preds = np.argmax(A2, axis=1)
    for i, shape in enumerate(SHAPES):
        mask = y_test == i
        if mask.sum() > 0:
            acc = np.mean(preds[mask] == i) * 100
            print(f"  {shape:<10}: {acc:.1f}%")


if __name__ == "__main__":
    train()