"""Tune the guidance strengths with TFG's beam search (accuracy on 64 images).

Start at 0.25 (gamma: 0.001). Each step multiplies one strength by 2 (gamma by 10), keeps the best 2 candidates,
at most 6 steps, as in the TFG repo. Strengths only go up. Seed 12345 (the final runs use seed 0).
  family "simple":      rho                    DPS, OU-DPS
  family "tfg":         rho, mu, gamma         TFG, OU-TFG
  family "mean":        mu, gamma (rho = 0)    mean-only
  family "tfg_then_ou": ou_rho, ou_mu of the last --ou_steps steps, TFG settings from --base (also tries 0, 0)

  python ou_guidance/search.py --method ou --family simple --label 8
Saves <method>_<family>_label<L>_guided<k>_n<n>[_ou<k>][_st].json: the best "settings" and every trial ("tried").
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from models import load_diffusion, load_guidance_classifier, load_eval_classifier
from sampler import sample
from evaluate import accuracy

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["network", "ou"], required=True)
p.add_argument("--family", choices=["simple", "tfg", "mean", "tfg_then_ou"], required=True)
p.add_argument("--label", type=int, default=8)
p.add_argument("--guided_steps", type=int, default=100)
p.add_argument("--ou_steps", type=int, default=0)       # TFG-then-OU: last steps with the OU guess
p.add_argument("--straight_through", action="store_true")  # ablation: the gradient passes the clip
p.add_argument("--base", default=None)                   # TFG-then-OU: the TFG search result (json)
p.add_argument("--n", type=int, default=64)              # images per trial
p.add_argument("--beam", type=int, default=2)            # candidates kept (TFG repo: 2)
p.add_argument("--max_steps", type=int, default=6)       # TFG repo: 6
p.add_argument("--steps", type=int, default=100)         # DDIM steps
p.add_argument("--classifier", default="checkpoints/resnet_cifar10.pt")
p.add_argument("--out", default="results/searches")
a = p.parse_args()

TUNED = {"simple": ["rho"], "tfg": ["rho", "mu", "gamma"], "mean": ["mu", "gamma"], "tfg_then_ou": ["ou_rho", "ou_mu"]}[a.family]
FIXED = {"simple": dict(mu=0.0, n_iter=0, gamma=0.0), "tfg": dict(n_iter=4),
         "mean": dict(rho=0.0, n_iter=4), "tfg_then_ou": {}}[a.family]
if a.family == "tfg_then_ou":
    FIXED = dict(json.load(open(a.base))["settings"], ou_steps=a.ou_steps)
if a.straight_through:
    FIXED["straight_through"] = True                       # passed on to sample() and run.py
START = {"rho": 0.25, "mu": 0.25, "gamma": 0.001, "ou_rho": 0.25, "ou_mu": 0.25}
FACTOR = {"rho": 2, "mu": 2, "gamma": 10, "ou_rho": 2, "ou_mu": 2}
MAXIMUM = {"rho": 8, "mu": 8, "gamma": 1, "ou_rho": 8, "ou_mu": 8}              # caps, as in the TFG repo

name = f"{a.method}_{a.family}_label{a.label}_guided{a.guided_steps}_n{a.n}" + (f"_ou{a.ou_steps}" if a.family == "tfg_then_ou" else "") + ("_st" if a.straight_through else "")
os.makedirs(a.out, exist_ok=True)
file = os.path.join(a.out, name + ".json")
tried = json.load(open(file))["tried"] if os.path.exists(file) else {}     # {settings: accuracy}, to resume

device = "cuda" if torch.cuda.is_available() else "cpu"
unet, scheduler = load_diffusion(device)
classifier = load_guidance_classifier(a.classifier, device)
processor, judge = load_eval_classifier(device)


def trial(config):
    """Accuracy of `config` (saved; the same noise for every config)."""
    key = json.dumps(config, sort_keys=True)
    if key not in tried:
        images, _ = sample(unet, scheduler, classifier, a.label, a.n, a.method, guided_steps=a.guided_steps,
                           steps=a.steps, seed=12345, device=device, **FIXED, **config)
        tried[key] = accuracy(images, a.label, processor, judge, device)
        print(f"  tried {key}: accuracy {tried[key]:.2f}", flush=True)
        json.dump({"tried": tried}, open(file, "w"), indent=1)
    return tried[key]


beam = [{p: START[p] for p in TUNED}]
if a.family == "tfg_then_ou":
    beam.append({p: 0.0 for p in TUNED})                                     # also try: no guidance in those steps
for step in range(a.max_steps):
    pool = list(beam)
    for c in beam:
        for p in TUNED:                                                      # increase one strength at a time
            new = dict(c, **{p: min(round(c[p] * FACTOR[p], 6), MAXIMUM[p])})
            if new not in pool:
                pool.append(new)
    new_beam = sorted(pool, key=trial, reverse=True)[:a.beam]               # keep the best
    print(f"step {step + 1}: best so far {new_beam[0]} (accuracy {trial(new_beam[0]):.2f})")
    if new_beam == beam:
        break                                                                # no change: stop
    beam = new_beam

best = beam[0]
result = {"best_accuracy": trial(best), "settings": {**FIXED, **best}, "tried": tried}   # settings can go to run.py
json.dump(result, open(file, "w"), indent=1)
print("BEST:", result["settings"], "accuracy", result["best_accuracy"])
