
# BACKPROP TRAINING ON SIMPLE SHAPE DATASET

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
#threshold = float(input("Enter threshold (e.g. 1.0): "))


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cpu"
NUM_CLASSES = 5

#load data
transform = transforms.Compose([
    transforms.Grayscale(),        
    transforms.Resize((28, 28)),   
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#data_path = r"D:\dataset\simple_shapes_off_centered_noisy"

train_data = datasets.ImageFolder(root=data_path + "\\train", transform=transform)
test_data  = datasets.ImageFolder(root=data_path + "\\test",  transform=transform)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

print(f"Train: {len(train_data)} | Test: {len(test_data)}")
print(f"Classes: {train_data.classes}")

## BP model

class BPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(784, 500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.ReLU(),

            nn.Linear(500, 500),
            nn.ReLU(),

            nn.Linear(500, NUM_CLASSES)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# set training setup

model = BPNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# train loop happens here

def train(model, loader, epochs=epochs):
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
        #print(f"[BP] Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Train Acc: {acc:.2f}%")
        test_acc = evaluate(model, test_loader)
        print(
            f"[BP] Epoch {epoch+1:02d} | "
            f"Loss: {total_loss:.4f} | "
            f"Train Acc: {acc:.2f}% | "
            f"Test Acc: {test_acc:.2f}%"
        )

# created evaulation func

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

    acc = 100 * correct / total
    #print(f"\n[BP] Test Accuracy: {acc:.2f}%")
    return acc

# code run trainning here
train(model, train_loader, epochs=epochs)
# code is testing model
evaluate(model, test_loader)
