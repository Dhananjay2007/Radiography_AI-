import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# ------------------------------
# CONFIGURATION
# ------------------------------
BASE_DIR = r"D:\COVID-19_Radiography_Dataset"
MODEL_PATH = r"D:\RadImageNet_pytorch\ResNet50.pt"  # optional pre-trained weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4

# ------------------------------
# DATASET CLASS
# ------------------------------
class COVIDDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        classes = ['COVID', 'Normal']

        for label, cls in enumerate(classes):
            img_dir = os.path.join(base_dir, cls, 'images')
            if not os.path.exists(img_dir):
                print(f"Skipping missing folder: {img_dir}")
                continue

            for file in os.listdir(img_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(img_dir, file))
                    self.labels.append(label)

        print(f"✅ Loaded {len(self.image_paths)} images from {len(classes)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


# ------------------------------
# MODEL SETUP
# ------------------------------
def load_resnet50(num_classes=2):
    model = models.resnet50(weights=None)
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict, strict=False)
            print("✅ RadImageNet weights loaded (partial).")
        except Exception as e:
            print("⚠️ Could not load RadImageNet weights:", e)

    # Replace final FC layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


# ------------------------------
# GRAD-CAM IMPLEMENTATION
# ------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target = dict(self.model.named_modules())[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()


# ------------------------------
# VISUALIZATION FUNCTION
# ------------------------------
def visualize_heatmap(image_path, cam, prediction, confidence):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    # Bounding box
    threshold = np.percentile(cam, 99)
    mask = np.uint8(cam > threshold) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color = (0, 255, 0) if prediction == "Normal" else (0, 0, 255)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

    # Save
    result_path = os.path.splitext(image_path)[0] + "_heatmap.jpg"
    cv2.putText(overlay, f"{prediction} ({confidence:.2f})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(result_path, overlay)
    print(f"🩻 Saved heatmap: {result_path}")
    return result_path


# ------------------------------
# DISEASE REPORT GENERATION
# ------------------------------
def generate_report(prediction, confidence):
    if prediction == "COVID":
        risk = "HIGH RISK" if confidence > 0.85 else "MODERATE RISK"
        summary = f"The scan indicates possible COVID-19 infection with {confidence*100:.1f}% confidence.\n" \
                  f"Risk Level: {risk}.\nImmediate medical evaluation is recommended."
    else:
        summary = f"The scan appears normal with {confidence*100:.1f}% confidence.\nNo visible signs of infection detected."

    print("\n🧾 --- Disease Summary Report ---")
    print(summary)
    print("--------------------------------\n")
    return summary


# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = COVIDDataset(BASE_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = load_resnet50(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --- TRAINING ---


    # --- TRAINING ---
    print("🚀 Training started...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", ncols=100)

        for imgs, labels, _ in progress_bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            avg_loss = total_loss / (total / BATCH_SIZE)

            progress_bar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "Accuracy": f"{acc:.2f}%"
            })

        print(f"✅ Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

    print("🎯 Training complete. Saving model...")
    torch.save(model.state_dict(), "covid_classifier.pth")
    print("✅ Model saved as covid_classifier.pth\n")

    # --- INFERENCE + HEATMAP ---
    model.eval()
    grad_cam = GradCAM(model, target_layer='layer4')

    test_img_path = dataset.image_paths[0]
    input_img = transform(Image.open(test_img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    output = model(input_img)
    probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
    pred_idx = np.argmax(probs)
    classes = ["COVID", "Normal"]
    prediction = classes[pred_idx]
    confidence = probs[pred_idx]

    cam = grad_cam.generate(input_img, class_idx=pred_idx)
    result_path = visualize_heatmap(test_img_path, cam, prediction, confidence)
    generate_report(prediction, confidence)
