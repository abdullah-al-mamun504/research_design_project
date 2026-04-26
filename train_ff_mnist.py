
# Experiment 3 : TRAIN  FF WITH MINIST dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cpu"

# dataset load
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_path = r"D:\Research design course\minist_dataset_experiment"

train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=256, shuffle=False)

print(f"Train: {len(train_data)} | Test: {len(test_data)}")
print(f"Classes: {train_data.classes}")

# helper one_hot helper for ff minist

def one_hot(y):
    return F.one_hot(y, 10).float()

def make_input(x, y):
    x = x.view(x.size(0), -1)
    y_oh = one_hot(y)

    # normalize label scale to match input
    y_oh = y_oh * 2 - 1   # convert to [-1, 1]

    x = x.clone()
    x[:, :10] = y_oh
    return x

def neg_labels(y):
    rand = torch.randint(0, 10, y.shape)
    mask = rand == y
    rand[mask] = (rand[mask] + 1) % 10
    return rand

# Forward forward layer

class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.003)
        self.threshold = 1.0

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x

    def train_step(self, x_pos, x_neg):
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        g_pos = (h_pos ** 2).mean(dim=1)
        g_neg = (h_neg ** 2).mean(dim=1)

        loss = F.softplus(-g_pos + self.threshold).mean() + \
               F.softplus(g_neg - self.threshold).mean()

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()

        return h_pos.detach(), h_neg.detach(), loss.item()

# FF net flow creation

class FFNet:
    def __init__(self):
        self.layers = [
            FFLayer(784, 200),
            FFLayer(200, 200),
            FFLayer(200, 200)
        ]

    def train(self, loader, epochs=20):
        for epoch in range(epochs):
            total_loss = 0

            for x, y in loader:
                y_neg = neg_labels(y)

                x_pos = make_input(x, y)
                x_neg = make_input(x, y_neg)

                for layer in self.layers:
                    x_pos, x_neg, loss = layer.train_step(x_pos, x_neg)
                    total_loss += loss

            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    def predict(self, x):
        x = x.view(x.size(0), -1)
        scores = []

        for label in range(10):
            y = torch.full((x.size(0),), label)
            x_l = make_input(x, y)

            h = x_l
            g = 0

            for layer in self.layers:
                h = layer.forward(h)
                g += (h ** 2).mean(dim=1)

            scores.append(g.unsqueeze(1))

        scores = torch.cat(scores, dim=1)
        return scores.argmax(dim=1)

# train plus test

net = FFNet()
net.train(train_loader, epochs=20)

correct, total = 0, 0

for x, y in test_loader:
    preds = net.predict(x)
    correct += (preds == y).sum().item()
    total += y.size(0)

print("Test Accuracy:", 100 * correct / total)


"""
Here i have note down few scope of improving the model for MINIST
1. self.layers is small 200 number, increase it to 500, could improve +1–2% accuracy
2. could have dynamic layer selection with input function 40+ epoch could improve 
3. model need more epoch 
4. 
"""

# =====================================================
# Experiment 3 : TRAIN WITH FF MINIST dataset
# ======================================================
