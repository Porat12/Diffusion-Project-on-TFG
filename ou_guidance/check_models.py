"""Accuracy of the guide and the judge on real CIFAR10 test images (both should be about 95%)."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import load_guidance_classifier, load_eval_classifier
from evaluate import load_cifar10

checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/resnet_cifar10.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
test_images, test_labels = load_cifar10("test")
images = torch.tensor(test_images).permute(0, 3, 1, 2).float() / 255          # (10000, 3, 32, 32) in [0, 1]
labels = torch.tensor(test_labels)

# ---- 1. guide (ResNet) ----
resnet = load_guidance_classifier(checkpoint, device)
with torch.no_grad():
    preds = torch.cat([resnet(images[i:i + 500].to(device)).argmax(1).cpu() for i in range(0, 10000, 500)])
print(f"ResNet accuracy on 10000 real test images: {(preds == labels).float().mean().item():.3f}")

# ---- 2. judge (ConvNeXt), the first 2000 test images ----
processor, convnext = load_eval_classifier(device)
n, correct = 2000, 0
with torch.no_grad():
    for i in range(0, n, 100):
        batch = [img for img in images[i:i + 100]]
        inputs = processor(batch, return_tensors="pt", do_rescale=False).to(device)
        correct += (convnext(**inputs).logits.argmax(1).cpu() == labels[i:i + 100]).sum().item()
print(f"ConvNeXt accuracy on {n} real test images: {correct / n:.3f}")
