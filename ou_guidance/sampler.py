"""Sampling with classifier guidance: a small version of TFG's Algorithm 1.

DDIM from t = 999 (noise) to t = 0 (image) in `steps` steps (eta = 1). Every step:
  1. the U-Net predicts the noise eps
  2. guess the clean image x0 (`method`, the only difference between the methods):
       "network": x0 = (x_t - sqrt(1 - ab) * eps) / sqrt(ab)   DPS / TFG, backprop through the U-Net
       "ou":      x0 = x_t / sqrt(ab)                          OU-DPS / OU-TFG, no backprop
  3. guidance with the gradient of log p(label | x0):
       rho         variance guidance (gradient w.r.t. x_t)
       mu, n_iter  mean guidance (n_iter small steps on x0 itself)
       gamma       a little noise on x0 before the classifier
  4. DDIM step
ab = alpha_bar_t, the signal left at step t. rho and mu follow TFG's "increase" schedule (average 1).
"""
import torch
import torch.nn.functional as F


def to_classifier_input(x):
    """[-1, 1] -> [0, 1] (the classifier normalises by itself)."""
    return (x + 1) / 2


def clean_guess(x, eps, ab, method):
    if method == "network":
        return (x - (1 - ab).sqrt() * eps) / ab.sqrt()
    if method == "ou":
        return x / ab.sqrt()
    raise ValueError(method)


def sampling_timesteps(steps):
    """The noise levels we visit, as in the TFG repo: [999, 989, ..., 10, 0] for 100 steps."""
    return [round(k * 999 / (steps - 1)) for k in range(steps)][::-1]


def clip(x0, straight_through=False):
    """Clip x0 to [-1, 1] (as the TFG repo does); clipped pixels get no gradient.
    straight_through: clip the values but keep the gradient (an ablation)."""
    clipped = x0.clamp(-1, 1)
    return x0 + (clipped - x0).detach() if straight_through else clipped


def sample(unet, scheduler, classifier, label, n, method="none", rho=0.0, guided_steps=100,
           mu=0.0, n_iter=0, gamma=0.0, steps=100, seed=0, device="cpu",
           ou_steps=0, ou_rho=None, ou_mu=None, straight_through=False):
    """Generate n images of `label`. Returns images in [0, 1] and counts of the expensive operations.
    method: "none", "network" or "ou". guided_steps: guide only the last k steps. rho = 0, mu > 0: mean-only.
    ou_steps = k (method "network"): TFG-then-OU, the last k steps use the "ou" guess with strengths ou_rho, ou_mu
    (default: rho, mu; both 0: no guidance there).
    seed: same seed = same starting noise for every method."""
    gen = torch.Generator(device).manual_seed(seed)
    alpha_bar = scheduler.alphas_cumprod.to(device)
    timesteps = sampling_timesteps(steps)
    x = torch.randn(n, 3, 32, 32, generator=gen, device=device)
    cost = {"unet_forward": 0, "unet_backward": 0, "classifier_calls": 0}

    # per step: signal left now (ab) and after the step (ab_prev), and the schedule's strength
    ab_all = alpha_bar[timesteps]
    ab_prev_all = torch.cat([ab_all[1:], torch.ones(1, device=device)])
    alpha_all = ab_all / ab_prev_all                                # alpha_t of the paper
    strength_all = alpha_all / alpha_all.mean()                     # "increase" schedule, average 1

    def log_prob(image, ab):
        """log p(label | image), with the smoothing noise gamma."""
        if gamma > 0:
            image = image + gamma * (1 - ab).sqrt() * torch.randn(image.shape, generator=gen, device=device)
        cost["classifier_calls"] += 1
        return F.log_softmax(classifier(to_classifier_input(image)), dim=1)[:, label].sum()

    for i, t in enumerate(timesteps):
        ab, ab_prev, alpha_t = ab_all[i], ab_prev_all[i], alpha_all[i]
        steps_left = len(timesteps) - i                             # 1 at the last step
        guided = method != "none" and steps_left <= guided_steps    # guide only the last steps
        guess, rho_now, mu_now = method, rho, mu
        if method == "network" and steps_left <= ou_steps:          # TFG-then-OU: OU guess at the end
            guess = "ou"
            rho_now = rho if ou_rho is None else ou_rho
            mu_now = mu if ou_mu is None else ou_mu
        strength = strength_all[i].item()
        rho_now, mu_now = strength * rho_now, strength * mu_now
        guided = guided and (rho_now > 0 or mu_now > 0)             # zero strength: a plain step

        # 1. noise prediction (keep the U-Net graph only if we backprop through it)
        backprop = guided and guess == "network" and rho_now > 0
        x = x.detach().requires_grad_(guided and rho_now > 0)
        with torch.set_grad_enabled(backprop):
            eps = unet(x, t).sample
        cost["unet_forward"] += 1
        cost["unet_backward"] += int(backprop)

        # 2. the network's guess, clipped as in the TFG repo (the DDIM step uses it in every method)
        x0_net = clean_guess(x, eps, ab, "network").clamp(-1, 1)

        # 3. guidance
        if guided:
            x0 = x0_net if guess == "network" else clip(clean_guess(x, eps, ab, "ou"), straight_through)
            grad = torch.autograd.grad(log_prob(x0, ab), x)[0] if rho_now > 0 else torch.zeros_like(x)  # variance guidance
            delta = torch.zeros_like(x0)                             # mean guidance
            for _ in range(n_iter):
                d = delta.detach().requires_grad_(True)
                delta = delta + mu_now * torch.autograd.grad(log_prob(x0.detach() + d, ab), d)[0]

        # 4. DDIM step from the network's clipped guess
        with torch.no_grad():
            x0_net = x0_net.detach()
            eps = (x - ab.sqrt() * x0_net) / (1 - ab).sqrt()         # the noise that matches the clipped guess
            sigma = ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
            x = ab_prev.sqrt() * x0_net + (1 - ab_prev - sigma ** 2).sqrt() * eps
            x = x + sigma * torch.randn(x.shape, generator=gen, device=device)
            if guided:
                x = x + rho_now * grad / alpha_t.sqrt() + ab_prev.sqrt() * delta.detach()

    return ((x.clamp(-1, 1) + 1) / 2).detach(), cost
