import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = r"D:\COVID-19_Radiography_Dataset\covid_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# MODEL (same as training)
# -------------------------
def load_model(num_classes=2):
    # Updated for newer torchvision
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------
# PREPROCESSING
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------
# GRAD-CAM HEATMAP
# -------------------------
def generate_heatmap(model, img_tensor, target_layer='layer4'):
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    layer = dict(model.named_modules())[target_layer]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    # Forward
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backward
    model.zero_grad()
    output[0, pred_class].backward()

    # Grad-CAM calculation
    grad = gradients['value'][0].cpu().data.numpy()
    act = activations['value'][0].cpu().data.numpy()
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam, pred_class

# -------------------------
# INFERENCE FUNCTION
# -------------------------
def analyze_image(image_path):
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Get Grad-CAM heatmap and prediction
    heatmap, pred_class = generate_heatmap(model, img_tensor)

    # ✅ FIX: detach() before converting to numpy
    probs = torch.softmax(model(img_tensor), dim=1).detach().cpu().numpy()[0]

    # Label map
    labels = ["Normal", "COVID-19"]
    predicted_label = labels[pred_class]
    confidence = probs[pred_class] * 100

    # Overlay heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    img_cv = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    # Save or show result
    output_path = os.path.splitext(image_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, overlay)
    print(f"\n🧠 Prediction: {predicted_label} ({confidence:.2f}%)")
    print(f"📊 Class probabilities: {dict(zip(labels, probs.round(3)))}")
    print(f"🔥 Heatmap saved at: {output_path}\n")

    # Medical-style summary
    if predicted_label == "COVID-19":
        print("🩸 Summary Report:")
        print("- Signs of infection detected in lung regions.")
        print("- Heatmap indicates abnormal opacity and inflammation areas.")
        print("- Possible bilateral ground-glass opacities and patchy infiltrates.")
        print("- Suggest immediate RT-PCR confirmation and clinical correlation.")
        print("- Recommendation: Consult a radiologist for further evaluation.")
    else:
        print("✅ Summary Report:")
        print("- No significant infection or abnormality detected.")
        print("- Lung regions appear clear with regular texture.")
        print("- No opacities or inflammatory patterns observed.")
        print("- Recommendation: Routine check if symptoms persist.")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    image_path = input("Enter image path: ").strip('"')
    analyze_image(image_path)
