# Handoff

## Latest verified state · 2026-11-03

The public README and experiment artifacts are available from the [repository](https://github.com/zzXavier/recurrent-diffusion-budget).

Verified experiment revision: `563754a2531a1e411a89c05b65eb0e38f48a9897`. A new GitHub checkout passed the unit tests, demo, 120 updates for each of five models and evaluation from saved predictions. The summary matched the archived values.

Next local command: `python run.py evaluate`.

## Earlier implementation notes


Current stage: local formal experiment and unit tests passed. Five trained model checkpoints, five held-out prediction files, individual generated samples, comparison image and summary are in `results/`. Run key: UCI-digits, seed 20261103, 120 updates, CPU.

Last verified commands: `python -m unittest discover -s tests`, `python run.py reproduce`. The formal run asserted a nonzero projection update and exact held-out prediction equality after checkpoint reload for every variant. The next independent check is `python run.py evaluate`, then reproduce into a fresh output directory and compare MSE and parameter counts against `results/summary.csv`.

Known limitations: one seed; low-resolution class conditions; no claim on natural-image generation or large-model performance.
