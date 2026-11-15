"""Run one experiment: generate images, evaluate them, save the result.

  python ou_guidance/run.py --method ou --label 8 --rho 2.0 --guided_steps 20 --n 100

Saves <name>.json (settings, accuracy, FID, time, memory, operation counts) and <name>.png (the first 64 images).
Skipped if the json already exists.
"""
import argparse, json, os, sys, time
import torch
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(__file__))
from models import load_diffusion, load_guidance_classifier, load_eval_classifier
from sampler import sample
from evaluate import accuracy, fid

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["none", "network", "ou"], required=True)  # none = unconditional
p.add_argument("--label", type=int, default=8)          # CIFAR10 class (8 = ship)
p.add_argument("--rho", type=float, default=0.0)        # variance guidance strength
p.add_argument("--mu", type=float, default=0.0)         # mean guidance step size
p.add_argument("--n_iter", type=int, default=0)         # mean guidance iterations
p.add_argument("--gamma", type=float, default=0.0)      # smoothing noise
p.add_argument("--guided_steps", type=int, default=100)  # guide only the last k steps (100 = all)
p.add_argument("--ou_steps", type=int, default=0)       # TFG-then-OU: OU guess in the last k steps
p.add_argument("--ou_rho", type=float, default=None)    # strength in those steps (default: --rho)
p.add_argument("--ou_mu", type=float, default=None)     # (default: --mu); both 0 = no guidance there
p.add_argument("--straight_through", action="store_true")  # ablation: the gradient passes the clip
p.add_argument("--n", type=int, default=100)            # number of images
p.add_argument("--batch", type=int, default=100)
p.add_argument("--steps", type=int, default=100)        # DDIM steps
p.add_argument("--seed", type=int, default=0)
p.add_argument("--classifier", default="checkpoints/resnet_cifar10.pt")
p.add_argument("--out", default="results")
p.add_argument("--no_fid", action="store_true")         # skip FID (quick tests)
a = p.parse_args()

name = (f"{a.method}_label{a.label}_rho{a.rho}_guided{a.guided_steps}_mu{a.mu}_iter{a.n_iter}_gamma{a.gamma}"
        f"_n{a.n}_seed{a.seed}" + (f"_ou{a.ou_steps}_ourho{a.ou_rho}_oumu{a.ou_mu}" if a.ou_steps else "")
        + ("_st" if a.straight_through else ""))
os.makedirs(a.out, exist_ok=True)
result_file = os.path.join(a.out, name + ".json")
if os.path.exists(result_file):
    print("already done:", result_file); sys.exit()

device = "cuda" if torch.cuda.is_available() else "cpu"
unet, scheduler = load_diffusion(device)
classifier = load_guidance_classifier(a.classifier, device)

# ---- generate ----
images, cost = [], {}
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
start = time.time()
for i in range(0, a.n, a.batch):                         # the seed does not depend on the method: same starting noise
    imgs, c = sample(unet, scheduler, classifier, a.label, min(a.batch, a.n - i), a.method, a.rho, a.guided_steps,
                     a.mu, a.n_iter, a.gamma, a.steps, seed=a.seed * 1000 + i, device=device,
                     ou_steps=a.ou_steps, ou_rho=a.ou_rho, ou_mu=a.ou_mu,
                     straight_through=a.straight_through)
    images.append(imgs)
    cost = {k: cost.get(k, 0) + v for k, v in c.items()}
if device == "cuda":
    torch.cuda.synchronize()
seconds = time.time() - start
images = torch.cat(images)

# ---- evaluate ----
processor, eval_model = load_eval_classifier(device)
result = dict(vars(a), accuracy=accuracy(images, a.label, processor, eval_model, device),
              seconds_per_image=seconds / a.n, **cost)
if device == "cuda":
    result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
if not a.no_fid:
    result["fid"] = fid(images, a.label, device)

save_image(images[:64], os.path.join(a.out, name + ".png"), nrow=8)   # the first 64 images, 8 x 8
json.dump(result, open(result_file, "w"), indent=1)
print(json.dumps(result, indent=1))
