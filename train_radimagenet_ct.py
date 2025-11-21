import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os

def main():
    # ==========================
    # CONFIG
    # ==========================
    train_dir = r"D:\head_ct\data\train"
    val_dir = r"D:\head_ct\data\val"
    model_path = r"D:\RadImageNet_pytorch\ResNet50.pt"
    num_classes = 2
    batch_size = 8
    num_epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================
    # TRANSFORMS
    # ==========================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # ==========================
    # DATASET & DATALOADER
    # ==========================
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ==========================
    # MODEL LOADING
    # ==========================
    print("Loading pretrained RadImageNet ResNet50 model...")
    model = models.resnet50()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ==========================
    # TRAINING SETUP
    # ==========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ==========================
    # TRAINING LOOP
    # ==========================
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Train Loss: {running_loss / total:.4f} | Train Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

    # ==========================
    # SAVE MODEL
    # ==========================
    save_path = r"D:\head_ct\models\resnet50_radimagenet_finetuned.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to: {save_path}")

if __name__ == "__main__":
    main()
