###########################################
# Forward Forward - Simple Shapes (Fixed)
###########################################

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ─────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────
print("1: Centered Shapes")
print("2: Not Centered Shapes")
print("3: Centered noisy Shapes")
print("4: Not Centered noisy Shapes")
choice = input("Select dataset (1–4): ")

if choice == '1':
    data_path = r"D:\dataset\simple_shapes_centered"
elif choice == '2':
    data_path = r"D:\dataset\simple_shapes"
elif choice == '3':
    data_path = r"D:\dataset\simple_shapes_centered_noisy"
else:
    data_path = r"D:\dataset\simple_shapes_off_centered_noisy"

epochs    = int(input("Enter number of epochs: "))
lr        = float(input("Enter learning rate: "))
threshold = float(input("Enter threshold: "))

NUM_CLASSES = 5

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root=data_path + "\\train", transform=transform)
test_data  = datasets.ImageFolder(root=data_path + "\\test",  transform=transform)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

print(f"Train: {len(train_data)} | Test: {len(test_data)}")
print(f"Classes: {train_data.classes}")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def make_input(x, y):
    x = x.view(x.size(0), -1)
    y_oh = F.one_hot(y, NUM_CLASSES).float() * 2 - 1
    x = x.clone()
    x[:, :NUM_CLASSES] = y_oh
    return x

# ✅ FIX 1: neg_labels — guaranteed different label via random offset
def neg_labels(y):
    offsets = torch.randint(1, NUM_CLASSES, y.shape)
    return (y + offsets) % NUM_CLASSES

def per_class_accuracy(net, loader):
    correct_per = [0] * NUM_CLASSES
    total_per   = [0] * NUM_CLASSES
    for x, y in loader:
        preds = net.predict(x)
        for c in range(NUM_CLASSES):
            mask = (y == c)
            correct_per[c] += (preds[mask] == y[mask]).sum().item()
            total_per[c]   += mask.sum().item()
    return [(correct_per[c], total_per[c]) for c in range(NUM_CLASSES)]

def confusion_matrix(net, loader):
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        preds = net.predict(x)
        for t, p in zip(y, preds):
            matrix[t][p] += 1
    return matrix

def print_confusion(matrix, class_names):
    print("\nConfusion matrix:")
    print(f"{'':12}" + "".join(f"{n:>10}" for n in class_names))
    for i, row in enumerate(matrix):
        print(f"{class_names[i]:<12}" + "".join(f"{v.item():>10}" for v in row))

def print_diagnostic_block(epoch, stored, class_names):
    print(f"\n[Epoch {epoch}]")
    print(f"  Loss: {stored['loss']:.2f} | Test Acc: {stored['acc']:.2f}%")

    print("\n  Per-class accuracy:")
    for i, name in enumerate(class_names):
        c, t = stored['per_class'][i]
        print(f"    {name:<12}: {100*c/max(t,1):.2f}% ({c}/{t})")

    print_confusion(stored['cm'], class_names)

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return torch.relu(self.linear(x))

    def train_step(self, xp, xn):
        hp = self.forward(xp)
        hn = self.forward(xn)

        gp = (hp ** 2).mean(dim=1)
        gn = (hn ** 2).mean(dim=1)

        loss = (F.softplus(-gp + threshold) +
                F.softplus( gn - threshold)).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # ✅ FIX 2: Normalize activations before passing to the next layer
        # Goodness is computed from raw h; normalized h is passed forward
        hp_norm = hp / (hp.norm(dim=1, keepdim=True) + 1e-8)
        hn_norm = hn / (hn.norm(dim=1, keepdim=True) + 1e-8)

        return hp_norm.detach(), hn_norm.detach(), loss.item()


class FFNet:
    def __init__(self):
        self.layers = [
            FFLayer(784, 500),
            FFLayer(500, 500),
            FFLayer(500, 500),
        ]

    def predict(self, x):
        x = x.view(x.size(0), -1)
        scores = []
        for c in range(NUM_CLASSES):
            y_c = torch.full((x.size(0),), c)
            xc  = make_input(x, y_c)
            h, g = xc, 0
            for layer in self.layers:
                h = layer.forward(h)
                g += (h ** 2).mean(dim=1)
                # ✅ FIX 2: Normalize between layers during inference too
                h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            scores.append(g.unsqueeze(1))
        return torch.cat(scores, dim=1).argmax(dim=1)

    def evaluate(self, loader):
        correct, total = 0, 0
        for x, y in loader:
            preds = self.predict(x)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total

    def train(self, loader, epochs):
        epoch_store = {}

        for epoch in range(1, epochs + 1):
            total_loss = 0

            for x, y in loader:
                y_neg = neg_labels(y)
                xp = make_input(x, y)
                xn = make_input(x, y_neg)

                for layer in self.layers:
                    xp, xn, loss = layer.train_step(xp, xn)
                    total_loss += loss

            test_acc  = self.evaluate(test_loader)
            train_acc = self.evaluate(loader)

            cm        = confusion_matrix(self, test_loader)
            per_class = per_class_accuracy(self, test_loader)

            epoch_store[epoch] = {
                'loss':      total_loss,
                'acc':       test_acc,
                'train_acc': train_acc,
                'cm':        cm,
                'per_class': per_class
            }

            print(f"Epoch {epoch} | Loss: {total_loss:.2f} | "
                  f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # ───────── DIAGNOSTICS ─────────
        print("\n===== FULL DIAGNOSTICS =====")
        print(f"Dataset Choice : {choice}")
        print(f"Epochs         : {epochs}")
        print(f"Learning Rate  : {lr}")
        print(f"Threshold      : {threshold}")
        print(f"Num Classes    : {NUM_CLASSES}")
        print(f"Classes        : {train_data.classes}")
        print(f"Train Samples  : {len(train_data)}")
        print(f"Test Samples   : {len(test_data)}")
        print(f"Architecture   : 784 → 500 → 500 → 500")
        print(f"Optimizer      : Adam")
        print(f"Batch Size     : 256")
        print(f"Loss Fn        : FF Softplus (layer-wise)")
        print(f"Normalization  : L2 inter-layer (fixed)")
        print(f"Neg Label Fn   : Offset-based (fixed)")
        print()

        for epoch in range(1, epochs + 1):
            print_diagnostic_block(epoch, epoch_store[epoch], train_data.classes)

        # ───────── FINAL SUMMARY ─────────
        best_epoch = max(epoch_store, key=lambda e: epoch_store[e]['acc'])
        best       = epoch_store[best_epoch]
        print(f"\n===== BEST EPOCH: {best_epoch} =====")
        print(f"  Train Acc : {best['train_acc']:.2f}%")
        print(f"  Test Acc  : {best['acc']:.2f}%")
        print(f"  Loss      : {best['loss']:.2f}")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
net = FFNet()
net.train(train_loader, epochs)
