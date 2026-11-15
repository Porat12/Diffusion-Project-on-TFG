# Figures

Made from the result files in `/content/drive/MyDrive/tfg_ou/results`. In every run the images start from the same random noise for every method (seed 0). Accuracy = fraction the ConvNeXt judge calls the target label (+- standard error); FID against the real CIFAR10 training images of that label (only comparable at equal n); seconds per image on the Colab GPU; U-Net backward passes per sampling run (100 steps).


## Label 8 (n = 100)

**label8_1_main_comparison.png**: every method with guidance at all 100 sampling steps. Accuracy (+- standard error), FID and seconds per image; the number of U-Net backward passes is next to each name. Hatched bars = straight-through ablation. Runs used:
- unconditional: no guidance  (`none_label8_rho0.0_guided100_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- DPS: rho=0.5  (`network_label8_rho0.5_guided100_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- TFG: rho=0.5, mu=0.25, gamma=0.001  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- mean-only: rho=0.0, mu=0.5, gamma=0.01  (`network_label8_rho0.0_guided100_mu0.5_iter4_gamma0.01_n100_seed0.json`)
- OU-DPS: rho=0.25  (`ou_label8_rho0.25_guided100_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001  (`ou_label8_rho0.25_guided100_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-DPS-ST: rho=0.25  (`ou_label8_rho0.25_guided100_mu0.0_iter0_gamma0.0_n100_seed0_st.json`)
- OU-TFG-ST: rho=0.25, mu=0.25, gamma=0.001  (`ou_label8_rho0.25_guided100_mu0.25_iter4_gamma0.001_n100_seed0_st.json`)

**label8_2_ou_guided_steps.png**: OU-DPS and OU-TFG (dashed and hollow = straight-through ablation) with guidance only in the last t* steps; strengths tuned once at t* = 100. Dotted lines = the other methods at all 100 steps. Runs used:
- OU-DPS: rho=0.25, t*=5  (`ou_label8_rho0.25_guided5_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-DPS: rho=0.25, t*=10  (`ou_label8_rho0.25_guided10_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-DPS: rho=0.25, t*=20  (`ou_label8_rho0.25_guided20_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-DPS: rho=0.25, t*=30  (`ou_label8_rho0.25_guided30_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-DPS: rho=0.25, t*=40  (`ou_label8_rho0.25_guided40_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-DPS: rho=0.25, t*=60  (`ou_label8_rho0.25_guided60_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-DPS: rho=0.25  (`ou_label8_rho0.25_guided100_mu0.0_iter0_gamma0.0_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001, t*=5  (`ou_label8_rho0.25_guided5_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001, t*=10  (`ou_label8_rho0.25_guided10_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001, t*=20  (`ou_label8_rho0.25_guided20_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001, t*=30  (`ou_label8_rho0.25_guided30_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001, t*=40  (`ou_label8_rho0.25_guided40_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001, t*=60  (`ou_label8_rho0.25_guided60_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-TFG: rho=0.25, mu=0.25, gamma=0.001  (`ou_label8_rho0.25_guided100_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- OU-DPS-ST: rho=0.25, t*=10  (`ou_label8_rho0.25_guided10_mu0.0_iter0_gamma0.0_n100_seed0_st.json`)
- OU-DPS-ST: rho=0.25, t*=30  (`ou_label8_rho0.25_guided30_mu0.0_iter0_gamma0.0_n100_seed0_st.json`)
- OU-DPS-ST: rho=0.25  (`ou_label8_rho0.25_guided100_mu0.0_iter0_gamma0.0_n100_seed0_st.json`)
- OU-TFG-ST: rho=0.25, mu=0.25, gamma=0.001, t*=10  (`ou_label8_rho0.25_guided10_mu0.25_iter4_gamma0.001_n100_seed0_st.json`)
- OU-TFG-ST: rho=0.25, mu=0.25, gamma=0.001, t*=30  (`ou_label8_rho0.25_guided30_mu0.25_iter4_gamma0.001_n100_seed0_st.json`)
- OU-TFG-ST: rho=0.25, mu=0.25, gamma=0.001  (`ou_label8_rho0.25_guided100_mu0.25_iter4_gamma0.001_n100_seed0_st.json`)

**label8_3_tfg_then_ou.png**: TFG with its tuned settings for the first 100 - k steps. TFG-then-OU: the last k steps use the OU guess with their own strengths (ou_rho, ou_mu, tuned by beam search for every k). TFG-then-unguided (the control): the last k steps have no guidance at all. If the two lines are equal, the OU guess adds nothing. Where the search found no OU strength better than none (ou_rho = ou_mu = 0), the TFG-then-OU run is the control itself and appears only on the control line. k = 0 is TFG. Runs used:
- TFG: rho=0.5, mu=0.25, gamma=0.001  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0.json`)
- TFG-then-OU: rho=0.5, mu=0.25, gamma=0.001, k=10, ou_rho=0.25, ou_mu=0.5  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou10_ourho0.25_oumu0.5.json`)
- TFG-then-OU: rho=0.5, mu=0.25, gamma=0.001, k=20, ou_rho=0.25, ou_mu=0.25  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou20_ourho0.25_oumu0.25.json`)
- TFG-then-OU: rho=0.5, mu=0.25, gamma=0.001, k=30, ou_rho=0.5, ou_mu=0.25  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou30_ourho0.5_oumu0.25.json`)
- TFG-then-OU: rho=0.5, mu=0.25, gamma=0.001, k=40, ou_rho=1.0, ou_mu=0.25  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou40_ourho1.0_oumu0.25.json`)
- TFG-then-OU: rho=0.5, mu=0.25, gamma=0.001, k=60, ou_rho=0.25, ou_mu=0.25  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou60_ourho0.25_oumu0.25.json`)
- TFG-then-unguided: rho=0.5, mu=0.25, gamma=0.001, k=5, ou_rho=0.0, ou_mu=0.0  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou5_ourho0.0_oumu0.0.json`)
- TFG-then-unguided: rho=0.5, mu=0.25, gamma=0.001, k=10, ou_rho=0.0, ou_mu=0.0  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou10_ourho0.0_oumu0.0.json`)
- TFG-then-unguided: rho=0.5, mu=0.25, gamma=0.001, k=20, ou_rho=0.0, ou_mu=0.0  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou20_ourho0.0_oumu0.0.json`)
- TFG-then-unguided: rho=0.5, mu=0.25, gamma=0.001, k=30, ou_rho=0.0, ou_mu=0.0  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou30_ourho0.0_oumu0.0.json`)
- TFG-then-unguided: rho=0.5, mu=0.25, gamma=0.001, k=40, ou_rho=0.0, ou_mu=0.0  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou40_ourho0.0_oumu0.0.json`)
- TFG-then-unguided: rho=0.5, mu=0.25, gamma=0.001, k=60, ou_rho=0.0, ou_mu=0.0  (`network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou60_ourho0.0_oumu0.0.json`)

**label8_4_accuracy_vs_time.png** and **label8_5_accuracy_vs_fid.png**: every run of this label as one point (hollow = straight-through ablation). Tags: t* = guided last steps, k = last steps with the OU guess. Up-left is best in the time plot; up-right in the FID plot (FID axis reversed).

**label8_6_samples.png**: the first 8 images of each run in figure 1. Each column starts from the same random noise, so differences within a column come from the guidance.

| experiment | settings | n | accuracy | FID | s/img | U-Net backward | file |
|---|---|---|---|---|---|---|---|
| unconditional | no guidance | 100 | 0.10 +- 0.03 | 221.4 | 0.26 | 0 | `none_label8_rho0.0_guided100_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| DPS | rho=0.5 | 100 | 0.90 +- 0.03 | 109.4 | 0.72 | 100 | `network_label8_rho0.5_guided100_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| TFG | rho=0.5, mu=0.25, gamma=0.001 | 100 | 0.85 +- 0.04 | 99.9 | 1.03 | 100 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| mean-only | rho=0.0, mu=0.5, gamma=0.01 | 100 | 0.34 +- 0.05 | 178.8 | 0.55 | 0 | `network_label8_rho0.0_guided100_mu0.5_iter4_gamma0.01_n100_seed0.json` |
| OU-DPS | rho=0.25 | 100 | 0.22 +- 0.04 | 212.7 | 0.33 | 0 | `ou_label8_rho0.25_guided100_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| OU-DPS | rho=0.25, t*=60 | 100 | 0.23 +- 0.04 | 205.8 | 0.32 | 0 | `ou_label8_rho0.25_guided60_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| OU-DPS | rho=0.25, t*=40 | 100 | 0.19 +- 0.04 | 209.5 | 0.30 | 0 | `ou_label8_rho0.25_guided40_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| OU-DPS | rho=0.25, t*=30 | 100 | 0.18 +- 0.04 | 208.6 | 0.30 | 0 | `ou_label8_rho0.25_guided30_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| OU-DPS | rho=0.25, t*=20 | 100 | 0.26 +- 0.04 | 204.8 | 0.28 | 0 | `ou_label8_rho0.25_guided20_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| OU-DPS | rho=0.25, t*=10 | 100 | 0.23 +- 0.04 | 209.0 | 0.28 | 0 | `ou_label8_rho0.25_guided10_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| OU-DPS | rho=0.25, t*=5 | 100 | 0.13 +- 0.03 | 222.9 | 0.28 | 0 | `ou_label8_rho0.25_guided5_mu0.0_iter0_gamma0.0_n100_seed0.json` |
| OU-TFG | rho=0.25, mu=0.25, gamma=0.001 | 100 | 0.17 +- 0.04 | 230.2 | 0.72 | 0 | `ou_label8_rho0.25_guided100_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| OU-TFG | rho=0.25, mu=0.25, gamma=0.001, t*=60 | 100 | 0.12 +- 0.03 | 230.9 | 0.57 | 0 | `ou_label8_rho0.25_guided60_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| OU-TFG | rho=0.25, mu=0.25, gamma=0.001, t*=40 | 100 | 0.19 +- 0.04 | 218.6 | 0.48 | 0 | `ou_label8_rho0.25_guided40_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| OU-TFG | rho=0.25, mu=0.25, gamma=0.001, t*=30 | 100 | 0.27 +- 0.04 | 213.2 | 0.44 | 0 | `ou_label8_rho0.25_guided30_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| OU-TFG | rho=0.25, mu=0.25, gamma=0.001, t*=20 | 100 | 0.21 +- 0.04 | 219.9 | 0.40 | 0 | `ou_label8_rho0.25_guided20_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| OU-TFG | rho=0.25, mu=0.25, gamma=0.001, t*=10 | 100 | 0.24 +- 0.04 | 209.7 | 0.36 | 0 | `ou_label8_rho0.25_guided10_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| OU-TFG | rho=0.25, mu=0.25, gamma=0.001, t*=5 | 100 | 0.16 +- 0.04 | 226.8 | 0.34 | 0 | `ou_label8_rho0.25_guided5_mu0.25_iter4_gamma0.001_n100_seed0.json` |
| OU-DPS-ST | rho=0.25 | 100 | 0.08 +- 0.03 | 280.1 | 0.35 | 0 | `ou_label8_rho0.25_guided100_mu0.0_iter0_gamma0.0_n100_seed0_st.json` |
| OU-DPS-ST | rho=0.25, t*=30 | 100 | 0.17 +- 0.04 | 209.3 | 0.33 | 0 | `ou_label8_rho0.25_guided30_mu0.0_iter0_gamma0.0_n100_seed0_st.json` |
| OU-DPS-ST | rho=0.25, t*=10 | 100 | 0.23 +- 0.04 | 208.2 | 0.31 | 0 | `ou_label8_rho0.25_guided10_mu0.0_iter0_gamma0.0_n100_seed0_st.json` |
| OU-TFG-ST | rho=0.25, mu=0.25, gamma=0.001 | 100 | 0.15 +- 0.04 | 290.4 | 0.70 | 0 | `ou_label8_rho0.25_guided100_mu0.25_iter4_gamma0.001_n100_seed0_st.json` |
| OU-TFG-ST | rho=0.25, mu=0.25, gamma=0.001, t*=30 | 100 | 0.20 +- 0.04 | 227.1 | 0.41 | 0 | `ou_label8_rho0.25_guided30_mu0.25_iter4_gamma0.001_n100_seed0_st.json` |
| OU-TFG-ST | rho=0.25, mu=0.25, gamma=0.001, t*=10 | 100 | 0.22 +- 0.04 | 209.5 | 0.34 | 0 | `ou_label8_rho0.25_guided10_mu0.25_iter4_gamma0.001_n100_seed0_st.json` |
| TFG-then-OU | rho=0.5, mu=0.25, gamma=0.001, k=10, ou_rho=0.25, ou_mu=0.5 | 100 | 0.87 +- 0.03 | 97.2 | 0.95 | 90 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou10_ourho0.25_oumu0.5.json` |
| TFG-then-OU | rho=0.5, mu=0.25, gamma=0.001, k=20, ou_rho=0.25, ou_mu=0.25 | 100 | 0.80 +- 0.04 | 101.6 | 0.92 | 80 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou20_ourho0.25_oumu0.25.json` |
| TFG-then-OU | rho=0.5, mu=0.25, gamma=0.001, k=30, ou_rho=0.5, ou_mu=0.25 | 100 | 0.79 +- 0.04 | 110.6 | 0.88 | 70 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou30_ourho0.5_oumu0.25.json` |
| TFG-then-OU | rho=0.5, mu=0.25, gamma=0.001, k=40, ou_rho=1.0, ou_mu=0.25 | 100 | 0.58 +- 0.05 | 137.9 | 0.85 | 60 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou40_ourho1.0_oumu0.25.json` |
| TFG-then-OU | rho=0.5, mu=0.25, gamma=0.001, k=60, ou_rho=0.25, ou_mu=0.25 | 100 | 0.39 +- 0.05 | 155.5 | 0.78 | 40 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou60_ourho0.25_oumu0.25.json` |
| TFG-then-unguided | rho=0.5, mu=0.25, gamma=0.001, k=5, ou_rho=0.0, ou_mu=0.0 | 100 | 0.84 +- 0.04 | 99.3 | 0.96 | 95 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou5_ourho0.0_oumu0.0.json` |
| TFG-then-unguided | rho=0.5, mu=0.25, gamma=0.001, k=10, ou_rho=0.0, ou_mu=0.0 | 100 | 0.81 +- 0.04 | 101.1 | 0.89 | 90 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou10_ourho0.0_oumu0.0.json` |
| TFG-then-unguided | rho=0.5, mu=0.25, gamma=0.001, k=20, ou_rho=0.0, ou_mu=0.0 | 100 | 0.80 +- 0.04 | 102.5 | 0.97 | 80 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou20_ourho0.0_oumu0.0.json` |
| TFG-then-unguided | rho=0.5, mu=0.25, gamma=0.001, k=30, ou_rho=0.0, ou_mu=0.0 | 100 | 0.70 +- 0.05 | 108.6 | 0.89 | 70 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou30_ourho0.0_oumu0.0.json` |
| TFG-then-unguided | rho=0.5, mu=0.25, gamma=0.001, k=40, ou_rho=0.0, ou_mu=0.0 | 100 | 0.58 +- 0.05 | 112.6 | 0.81 | 60 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou40_ourho0.0_oumu0.0.json` |
| TFG-then-unguided | rho=0.5, mu=0.25, gamma=0.001, k=60, ou_rho=0.0, ou_mu=0.0 | 100 | 0.28 +- 0.04 | 163.4 | 0.65 | 40 | `network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0_ou60_ourho0.0_oumu0.0.json` |