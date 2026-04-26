
# Simple Image Main Experiment 
# EXPERIMENT with Centered Shapes, Not Centered Shapes, Centered noisy Shapes, Not Centered noisy Shapes



print("1: Centered Shapes")
print("2: Not Centered Shapes")
print("3: Centered noisy Shapes")
print("4: Not Centered noisy Shapes")
choice = input("Select dataset (1 or 2): ")

if choice == '1':
    data_path = r"D:\dataset\simple_shapes_centered"
    print("--- Running Experiment: Centered Shapes ---")
elif choice == '2':
    data_path = r"D:\dataset\simple_shapes"
    print("--- Running Experiment: Not Centered Shapes ---")
elif choice == '3':
    data_path = r"D:\dataset\simple_shapes_centered_noisy"
    print("--- Running Experiment: Noisy Centered Shapes ---")
else:
    data_path = r"D:\dataset\simple_shapes_off_centered_noisy"
    print("--- Running Experiment: Not Centered Noisy Shapes ---")
    

epochs = int(input("Enter number of epochs: "))
lr = float(input("Enter learning rate (e.g. 0.003): "))
threshold = float(input("Enter threshold (e.g. 1.0): "))



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_CLASSES = 5

transform = transforms.Compose([
    transforms.Grayscale(),    # safety (in case RGB exists)
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#data_path = r"D:\dataset\simple_shapes"


train_data = datasets.ImageFolder(root=data_path + "\\train", transform=transform)
test_data  = datasets.ImageFolder(root=data_path + "\\test",  transform=transform)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

print(f"Train: {len(train_data)} | Test: {len(test_data)}")
print(f"Classes: {train_data.classes}")

def make_input(x, y):
    x    = x.view(x.size(0), -1)
    y_oh = F.one_hot(y, NUM_CLASSES).float() * 2 - 1
    x    = x.clone()
    x[:, :NUM_CLASSES] = y_oh
    return x

def neg_labels(y):
    r    = torch.randint(0, NUM_CLASSES, y.shape)
    mask = (r == y)
    r[mask] = (r[mask] + 1) % NUM_CLASSES
    return r

class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear    = nn.Linear(in_dim, out_dim)
        self.opt       = torch.optim.Adam(self.parameters(), lr=lr)
        self.threshold = threshold

    def forward(self, x):
        return torch.relu(self.linear(x))

    def train_step(self, xp, xn):
        hp = self.forward(xp)
        hn = self.forward(xn)
        gp = (hp ** 2).mean(dim=1)
        gn = (hn ** 2).mean(dim=1)
        loss = (F.softplus(-gp + self.threshold) +
                F.softplus( gn - self.threshold)).mean()
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()
        return hp.detach(), hn.detach(), loss.item()

class FFNet:
    def __init__(self):
        self.layers = [
            FFLayer(784, 500),
            FFLayer(500, 500),
            FFLayer(500, 500),
        ]

    def train(self, loader, epochs=epochs):
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for x, y in loader:
                y_neg = neg_labels(y)
                xp    = make_input(x, y)
                xn    = make_input(x, y_neg)
                for layer in self.layers:
                    xp, xn, loss = layer.train_step(xp, xn)
                    total_loss  += loss
            acc = self.evaluate(test_loader)
            print(f"Epoch {epoch:>3} | Loss: {total_loss:.2f} | Test Acc: {acc:.2f}%")

    def predict(self, x):
        x = x.view(x.size(0), -1)
        scores = []
        for c in range(NUM_CLASSES):
            y_c = torch.full((x.size(0),), c)
            xc  = make_input(x, y_c)
            h, g = xc, 0
            for layer in self.layers:
                h  = layer.forward(h)
                g += (h ** 2).mean(dim=1)
            scores.append(g.unsqueeze(1))
        return torch.cat(scores, dim=1).argmax(dim=1)

    def evaluate(self, loader):
        correct, total = 0, 0
        for x, y in loader:
            preds    = self.predict(x)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total

net = FFNet()
net.train(train_loader, epochs=epochs)
print(f"\nFinal Test Accuracy: {net.evaluate(test_loader):.2f}%")
