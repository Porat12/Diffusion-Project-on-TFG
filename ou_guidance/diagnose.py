"""How good is each clean-image guess during sampling? (sampling without guidance)

At every step, for the network guess and the OU guess x_t / sqrt(ab), with the guide:
  1. size of the guidance gradient
  2. fraction of pixels outside [-1, 1] (clipped: no gradient)
  3. agreement with the class of the finished image
  4. distance between the two guesses

  python ou_guidance/diagnose.py --n 64      (saves a json and a figure)
"""
import argparse, json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from models import load_diffusion, load_guidance_classifier
from sampler import clean_guess, to_classifier_input, sampling_timesteps

p = argparse.ArgumentParser()
p.add_argument("--label", type=int, default=8)
p.add_argument("--n", type=int, default=64)
p.add_argument("--steps", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--classifier", default="checkpoints/resnet_cifar10.pt")
p.add_argument("--out", default="results/diagnostics")
a = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
unet, scheduler = load_diffusion(device)
classifier = load_guidance_classifier(a.classifier, device)
alpha_bar = scheduler.alphas_cumprod.to(device)
timesteps = sampling_timesteps(a.steps)
gen = torch.Generator(device).manual_seed(a.seed)
x = torch.randn(a.n, 3, 32, 32, generator=gen, device=device)


def gradient_size(guess, x_t):
    """Mean norm of the gradient of log p(label | guess) w.r.t. x_t."""
    log_prob = F.log_softmax(classifier(to_classifier_input(guess.clamp(-1, 1))), dim=1)[:, a.label].sum()
    grad = torch.autograd.grad(log_prob, x_t, retain_graph=True)[0]
    return grad.flatten(1).norm(dim=1).mean().item()


rows, predictions = [], {"network": [], "ou": []}
for i, t in enumerate(timesteps):
    ab = alpha_bar[t]
    ab_prev = alpha_bar[timesteps[i + 1]] if t > 0 else torch.tensor(1.0, device=device)
    x = x.detach().requires_grad_(True)
    eps = unet(x, t).sample                                      # graph kept: the network guess is differentiated
    guess = {"network": clean_guess(x, eps, ab, "network"), "ou": clean_guess(x, eps.detach(), ab, "ou")}

    row = {"t": t, "distance_between_guesses": (guess["ou"] - guess["network"]).pow(2).mean().sqrt().item()}
    for name, g in guess.items():
        row[f"{name}_gradient"] = gradient_size(g, x)
        row[f"{name}_clipped"] = (g.detach().abs() > 1).float().mean().item()
        with torch.no_grad():
            predictions[name].append(classifier(to_classifier_input(g.clamp(-1, 1))).argmax(1))
    rows.append(row)

    with torch.no_grad():                                        # plain DDIM step
        x0_net = guess["network"].detach().clamp(-1, 1)          # clipped, as in the sampler
        eps = (x.detach() - ab.sqrt() * x0_net) / (1 - ab).sqrt()
        sigma = ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
        x = ab_prev.sqrt() * x0_net + (1 - ab_prev - sigma ** 2).sqrt() * eps
        x = x + sigma * torch.randn(x.shape, generator=gen, device=device)

with torch.no_grad():
    final = classifier(to_classifier_input(x.clamp(-1, 1))).argmax(1)         # class of the finished image
for name in predictions:
    for row, pred in zip(rows, predictions[name]):
        row[f"{name}_agreement"] = (pred == final).float().mean().item()

os.makedirs(a.out, exist_ok=True)
json.dump(rows, open(os.path.join(a.out, f"diagnose_label{a.label}.json"), "w"), indent=1)

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = [r["t"] for r in rows]
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
panels = [("Guidance gradient size (log scale)", "gradient", True),
          ("Fraction of pixels clipped (no gradient)", "clipped", False),
          ("Agreement with the final class", "agreement", False)]
for axis, (title, key, log) in zip(axes, panels):
    axis.plot(t, [r[f"network_{key}"] for r in rows], label="network guess (DPS / TFG)")
    axis.plot(t, [r[f"ou_{key}"] for r in rows], label="OU guess (analytic)")
    axis.set_title(title)
    if log:
        axis.set_yscale("log")
axes[3].plot(t, [r["distance_between_guesses"] for r in rows])
axes[3].set_title("Distance between the two guesses (RMS)")
axes[3].set_yscale("log")
for axis in axes:
    axis.invert_xaxis()                                          # noise (t = 999) on the left
    axis.set_xlabel("noise level t  (sampling goes left to right)")
axes[0].legend()
plt.tight_layout()
plt.savefig(os.path.join(a.out, f"diagnose_label{a.label}.png"), dpi=120)
print("saved to", a.out)
