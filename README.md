# Training-free guidance with the OU guess

Final project, course 00960236 "Generative Learning and Diffusion Models", Technion, Spring 2026.

Can TFG (Ye et al., NeurIPS 2024) guide a diffusion model without a backward pass through the network?
The idea: replace the network's clean-image estimate with the OU guess, which needs no network at all.
Task: CIFAR-10 label guidance, label ship.

- Report: [report/report.pdf](report/report.pdf)
- Slides: [slides/TFG_presentation.pdf](slides/TFG_presentation.pdf)

## The idea

Guidance needs an estimate of the clean image `x0` from the noisy `x_t` (`abar` = ᾱ_t):

| estimate | formula | cost |
|---|---|---|
| the network's (DPS, TFG) | `x0 = (x_t - sqrt(1 - abar) * eps_theta(x_t)) / sqrt(abar)` (Tweedie) | a backward pass through the U-Net |
| the OU guess | `x0 = x_t / sqrt(abar)` | no backward pass |

Everything else is TFG: variance guidance (`rho`), mean guidance (`mu`), smoothing (`gamma`), and its beam search
for the strengths.

## Results (label ship, 100 images per run)

| method | estimate | accuracy | time per image | peak memory |
|---|---|---|---|---|
| no guidance | - | 10% | 0.26 s | 1.6 GB |
| DPS | network | 90% | 0.72 s | 5.3 GB |
| TFG | network | 85% | 1.03 s | 5.3 GB |
| mean guidance only | network | 34% | 0.55 s | 1.6 GB |
| OU-DPS | OU | 22% | 0.33 s | 1.6 GB |
| OU-TFG | OU | 17% | 0.72 s | 1.6 GB |

- The OU guess is about as cheap as sampling without guidance, but it barely guides.
- It shows the class only in the last steps, after the class is already decided. Guiding only the last t* steps
  gives 12-27% for every t*.
- It fools the guide: the guide calls 97-100% of the OU images ship, the independent judge 17-20%.
- TFG, then the OU guess in the last k steps: up to k = 20 it keeps TFG's accuracy, but so does no guidance in those
  steps. The OU guess adds 5 points on average, within the noise.

## Files

| path | what |
|---|---|
| `report/` | the report (`report.tex`, `report.pdf`), its figures, and `make_figures.py`, which draws them |
| `slides/` | the slides (PDF) |
| `ou_guidance/` | the code (next table) |
| `notebooks/` | one Colab notebook per experiment |
| `results/` | per run: a `.json` (settings, accuracy, FID, time, memory) and a `.png` (the images); `searches/`: beam search logs; `diagnostics/`: the two estimates step by step |
| `results_checks/` | early checks, 32 images per run (from `01_checks`) |
| `figures/` | figures and the table of all runs (`results_table.csv`), from `09_plots`; `figures/README.md` lists the runs behind each figure |

| file in `ou_guidance/` | what |
|---|---|
| `sampler.py` | sampling with guidance (TFG Algorithm 1), with either estimate |
| `models.py` | the diffusion model, the guide (ResNet-18) and the judge (ConvNeXt-tiny) |
| `run.py` | one experiment: images, accuracy, FID, time, memory |
| `search.py` | TFG's beam search for the strengths |
| `evaluate.py` | accuracy and FID |
| `diagnose.py` | the two estimates step by step, during sampling without guidance |
| `guide_vs_judge.py` | the same images scored by the guide and by the judge |
| `plots.py` | the figures in `figures/` |
| `helpers.py` | functions for the notebooks |
| `check_models.py` | accuracy of both classifiers on real CIFAR-10 images |

## Experiments

| notebook | what | report section |
|---|---|---|
| `01_checks` | tests and timing | - |
| `01b_diagnostics` | when each estimate shows the class | 5.2 |
| `02_unconditional` | no guidance | 5.1 |
| `03_dps`, `04_tfg` | the baselines | 5.1 |
| `04b_mean_only` | mean guidance only | 5.1 |
| `05_ou_dps`, `06_ou_tfg` | the OU guess, at every step and in the last t* steps | 5.1, 5.2 |
| `07_tfg_then_ou`, `07b_tfg_then_unguided` | TFG, then the OU guess (or, as a control, no guidance) in the last k steps | 5.4 |
| `08_straight_through` | ablation: the gradient passes the clip | - |
| `09_plots` | the figures in `figures/` | - |
| `10_guide_vs_judge` | is the guide fooled? | 5.3 |

## How to run

**The experiments (Colab).** Put `ou_guidance/` and `resnet_cifar10.pt` (the guide's weights, from the TFG
repository) in `My Drive/tfg_ou/`, and run the notebooks on a T4 GPU in order. `02`-`06` can run in parallel. Each
notebook installs its packages, and finished runs are skipped.

**A quick local test** (the guide's weights in `checkpoints/`):

```
pip install -r requirements.txt
python ou_guidance/run.py --method network --rho 0.5 --n 4 --steps 10 --no_fid
```

**The report's figures**, from the result files:

```
python report/make_figures.py
```

## Notes

- Every run starts from the same noise (seed 0). The strengths were searched on 64 other images (seed 12345), scored
  by the judge, as in the paper.
- FID with 100 images is strongly biased: it is in the result files but not in the report.
- Time varies about 10% between Colab sessions.
- The diffusion model `google/ddpm-cifar10-32` has the noise schedule of the TFG repository, but a different checkpoint.
