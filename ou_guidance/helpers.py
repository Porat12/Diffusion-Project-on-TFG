"""Functions the notebooks use: run, search, diagnose, kind, show_results."""
import glob, json, os, subprocess, sys

ROOT = "/content/drive/MyDrive/tfg_ou"    # set by the notebooks


def _call(command):
    """Run a command and print its output live."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    if process.wait() != 0:
        raise RuntimeError("command failed: " + " ".join(command))


def run(method, label=8, n=100, out="results", **settings):
    """One experiment (run.py); finished runs are skipped. settings: rho, mu, n_iter, gamma, guided_steps, ..."""
    command = [sys.executable, f"{ROOT}/ou_guidance/run.py", "--method", method, "--label", str(label), "--n", str(n),
               "--classifier", f"{ROOT}/resnet_cifar10.pt", "--out", f"{ROOT}/{out}"]
    for name, value in settings.items():
        command += [f"--{name}"] if value is True else [f"--{name}", str(value)]   # True = a flag
    _call(command)


def search(method, family, label=8, guided_steps=100, n=64, max_steps=6, steps=100, ou_steps=0, straight_through=False):
    """Beam search (search.py); returns the best settings.
    family: "simple" (rho), "tfg" (rho, mu, gamma), "mean" (mu, gamma), "tfg_then_ou" (ou_rho, ou_mu; after the TFG search).
    straight_through: the ablation of notebook 08."""
    command = [sys.executable, f"{ROOT}/ou_guidance/search.py", "--method", method, "--family", family,
               "--label", str(label), "--guided_steps", str(guided_steps), "--n", str(n), "--max_steps", str(max_steps), "--steps", str(steps),
               "--classifier", f"{ROOT}/resnet_cifar10.pt", "--out", f"{ROOT}/results/searches"]
    suffix = ""
    if family == "tfg_then_ou":
        command += ["--ou_steps", str(ou_steps), "--base", f"{ROOT}/results/searches/network_tfg_label{label}_guided100_n{n}.json"]
        suffix = f"_ou{ou_steps}"
    if straight_through:
        command += ["--straight_through"]
        suffix += "_st"
    _call(command)
    file = f"{ROOT}/results/searches/{method}_{family}_label{label}_guided{guided_steps}_n{n}{suffix}.json"
    return json.load(open(file))["settings"]   # e.g. {'rho': 0.5, 'mu': 0.25, 'gamma': 0.001, 'n_iter': 4}


def diagnose(label=8, n=64):
    """Compare the two clean-image guesses step by step (diagnose.py); returns the figure's path."""
    _call([sys.executable, f"{ROOT}/ou_guidance/diagnose.py", "--label", str(label), "--n", str(n),
           "--classifier", f"{ROOT}/resnet_cifar10.pt", "--out", f"{ROOT}/results/diagnostics"])
    return f"{ROOT}/results/diagnostics/diagnose_label{label}.png"


def kind(r):
    """The experiment name of a result."""
    if r["method"] == "none":
        return "unconditional"
    extras = r["n_iter"] > 0
    st = "-ST" if r.get("straight_through") else ""      # straight-through ablation
    if r["method"] == "network" and r.get("ou_steps", 0) > 0:
        if r.get("ou_rho") == 0 and r.get("ou_mu") == 0:
            return "TFG-then-unguided"               # control: no guidance in the last steps
        return "TFG-then-OU" + st                    # OU guess in the last steps
    if r["method"] == "network":
        if r["rho"] == 0:
            return "mean-only"                      # no U-Net backprop
        return "TFG" if extras else "DPS"
    return ("OU-TFG" if extras else "OU-DPS") + st   # the analytic (OU) guess


def show_results(label=None, kinds=None, out="results"):
    """Print one line per run (kinds: only these experiments, e.g. ["DPS", "OU-DPS"])."""
    rows = [json.load(open(f)) for f in glob.glob(f"{ROOT}/{out}/*.json")]
    rows = [r for r in rows if (label is None or r["label"] == label) and (kinds is None or kind(r) in kinds)]
    rows.sort(key=lambda r: (r["label"], kind(r), -r["guided_steps"], r.get("ou_steps", 0)))
    print(f"{'experiment':18s} label guided ou_steps   rho    mu  gamma  ou_rho ou_mu |   n  accuracy (+-)     FID | s/img  unet_backward")
    for r in rows:
        error = (r["accuracy"] * (1 - r["accuracy"]) / r["n"]) ** 0.5       # standard error
        print(f"{kind(r):18s} {r['label']:5d} {r['guided_steps']:6d} {r.get('ou_steps', 0):7d} {r['rho']:5} {r['mu']:5} {r['gamma']:6} {str(r.get('ou_rho', '')):>6} {str(r.get('ou_mu', '')):>5} | "
              f"{r['n']:3d}  {r['accuracy']:6.2f} +-{error:.2f} {r.get('fid', float('nan')):6.1f} | {r['seconds_per_image']:5.2f}  {r['unet_backward']}")
