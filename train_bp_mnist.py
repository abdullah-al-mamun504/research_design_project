
# Experiment 4 : TRAIN Backdrop with MINIST dataset



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cpu"


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


#  BPackprop Model

class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(),
            
            nn.Linear(200, 200),
            nn.ReLU(),
            
            nn.Linear(200, 200),
            nn.ReLU(),
            
            nn.Linear(200, 10)  # output layer
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# Seting up my BP  Training

model = BPNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# looping the trainning
def train(model, loader, epochs=20):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Train Acc: {acc:.2f}%")


def evaluate(model, loader):
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = outputs.argmax(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

#running training BP model
train(model, train_loader, epochs=20)
# testing here
evaluate(model, test_loader)
