"""Accuracy (the right class?) and FID (do they look real?) of generated images."""
import io
import numpy as np
import torch
from torch.utils.data import Dataset


def load_cifar10(split, label=None):
    """Real CIFAR10 images (uint8, N x 32 x 32 x 3) and labels from the Hugging Face hub. label: only that class."""
    import pandas as pd
    from PIL import Image
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("uoft-cs/cifar10", f"plain_text/{split}-00000-of-00001.parquet", repo_type="dataset")
    table = pd.read_parquet(path)
    if label is not None:
        table = table[table["label"] == label]
    images = np.stack([np.array(Image.open(io.BytesIO(item["bytes"])).convert("RGB")) for item in table["img"]])
    return images, table["label"].to_numpy()


@torch.no_grad()
def accuracy(images, label, processor, model, device):
    """Fraction of the images the judge calls `label`. images: (N, 3, 32, 32) in [0, 1]."""
    correct = 0
    for i in range(0, len(images), 100):
        batch = [img for img in images[i:i + 100].cpu()]
        inputs = processor(batch, return_tensors="pt", do_rescale=False).to(device)  # resize + normalise
        correct += (model(**inputs).logits.argmax(1) == label).sum().item()
    return correct / len(images)


class _Images(Dataset):
    """uint8 images (N, 3, 32, 32) for torch-fidelity."""
    def __init__(self, uint8_images):
        self.x = uint8_images
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i]


def fid(images, label, device):
    """FID against the real training images of `label` (lower = better). Biased with few images: compare equal n only."""
    import torch_fidelity
    train_images, _ = load_cifar10("train", label)
    real = torch.tensor(train_images).permute(0, 3, 1, 2)                                     # uint8
    fake = (images.cpu() * 255).round().to(torch.uint8)
    out = torch_fidelity.calculate_metrics(
        input1=_Images(fake), input2=_Images(real), fid=True,
        input2_cache_name=f"cifar10_train_class{label}",   # real-image statistics are computed once
        cuda=device == "cuda", verbose=False,
        save_cpu_ram=True)                                  # no worker processes (safer on Colab / Windows)
    return out["frechet_inception_distance"]
