"""The three models: the diffusion model, the guide (steers sampling) and the judge (evaluates the images)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- guide: CIFAR ResNet18 with the layer names of TFG's checkpoint ----
# Input in [0, 1], normalised inside (95% on real test images, see check_models.py).
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        widths, strides, cin = [64, 128, 256, 512], [1, 2, 2, 2], 64
        for i, (w, s) in enumerate(zip(widths, strides), start=1):
            setattr(self, f"layer{i}", nn.Sequential(BasicBlock(cin, w, s), BasicBlock(w, w, 1)))
            cin = w
        self.fc = nn.Linear(512, num_classes)
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):  # images in [0, 1] -> class logits
        x = (x - self.mean) / self.std
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def load_guidance_classifier(path, device):
    """The guide: TFG's resnet_cifar10.pt."""
    net = ResNet18()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device).eval().requires_grad_(False)  # frozen: we only need gradients w.r.t. the image


# ---- diffusion model ----
def load_diffusion(device):
    """DDPM for CIFAR10 (the same noise schedule as the TFG repo's model)."""
    from diffusers import DDPMScheduler, UNet2DModel
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device).eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")  # holds alpha_bar
    return unet, scheduler


# ---- judge ----
def load_eval_classifier(device):
    """The judge: ConvNeXt-tiny for CIFAR10, as in the TFG repo. Never used for guidance."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    name = "ahsanjavid/convnext-tiny-finetuned-cifar10"
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name).to(device).eval()
    return processor, model
