import matplotlib
matplotlib.use("TkAgg")
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = r"D:\head_ct\models\resnet50_radimagenet_finetuned.pth"
TEST_IMAGE_PATH = r"D:\head_ct\data\val\normal\104.png"  # Change this
OUTPUT_DIR = r"D:\head_ct\outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# PREPROCESSING
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # Normal vs Hemorrhage (example)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# GRAD-CAM IMPLEMENTATION
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, target_class=None):
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, target_class, torch.softmax(output, dim=1)[0][target_class].item()

# -----------------------------
# REPORT GENERATOR
# -----------------------------
def generate_report(pred_label, confidence):
    if pred_label == "Hemorrhage":
        return (f"⚠️ The CT scan indicates possible intracranial hemorrhage or abnormal tissue density. "
                f"Immediate medical review is advised. Confidence: {confidence:.2f}%")
    else:
        return (f"✅ The CT scan appears normal with no significant hemorrhagic regions detected. "
                f"Confidence: {confidence:.2f}%")

# -----------------------------
# HEATMAP + PREDICTION
# -----------------------------
def analyze_ct_image(image_path, model):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    gradcam = GradCAM(model, model.layer4[-1])
    heatmap, class_idx, conf = gradcam.generate(input_tensor)

    # Map class index to label
    labels = ["Normal", "Hemorrhage"]
    pred_label = labels[class_idx]
    confidence = conf * 100

    # Apply heatmap
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    # Bounding boxes
    mask = (heatmap_resized > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        score = heatmap_resized[y:y+h, x:x+w].mean()
        color = (0, 0, 255) if score > 0.75 else (0, 255, 255) if score > 0.55 else (0, 255, 0)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
        cv2.putText(overlay, f"{score:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save
    output_heatmap = os.path.join(OUTPUT_DIR, "heatmap_overlay.jpg")
    cv2.imwrite(output_heatmap, overlay)
    print(f"✅ Saved visualization: {output_heatmap}")

    # Display summary
    report = generate_report(pred_label, confidence)
    print("\n=======================")
    print(f"Predicted Disease: {pred_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Report: {report}")
    print("=======================\n")

    # Optional: display preview
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    return pred_label, confidence, report, output_heatmap

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    pred, conf, rep, heatmap_path = analyze_ct_image(TEST_IMAGE_PATH, model)
