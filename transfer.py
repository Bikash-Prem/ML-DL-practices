import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --------------------------------------------------
# Classes
# --------------------------------------------------
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# --------------------------------------------------
# Transforms (ImageNet style)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # convert grayscale → 3 channels
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Dataset
# --------------------------------------------------
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=False
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=False
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --------------------------------------------------
# Model: Pretrained ResNet18
# --------------------------------------------------
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 10)

# Unfreeze last block for better fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

# --------------------------------------------------
# Loss & Optimizer
# --------------------------------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# --------------------------------------------------
# Training
# --------------------------------------------------
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# --------------------------------------------------
# Real Image Prediction
# --------------------------------------------------
def predict_real_image(image_path):
    model.eval()

    image = Image.open(image_path).convert("RGB")

    transform_real = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transform_real(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)

    return classes[prediction.item()], confidence.item()


label, conf = predict_real_image("shirt.jpg")
print(f"Prediction: {label}, Confidence: {conf*100:.2f}%")