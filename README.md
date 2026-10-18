# HAA-PR

Implementation of **Harmonic-ambiguity-aware pulse-rate measurement under motion.**

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0, pandas 2.2.3. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Use synchronized PPG/reference data such as [BIDMC](https://physionet.org/content/bidmc/1.0.0/) and the paper's exercise recordings. Input CSV columns are `time,ppg,reference,subject,record,split`; optional `ax,ay,az` add motion. Roles are `train,cal,test`, disjoint by subject. Resample each record to `sample_rate` before import. Reference rate is aligned to the same trailing analysis window. The CSV reference is an ECG-derived operational target; PPG-derived labels are not an independent reference.

```bash
python -m haa_pr prepare --input data/ppg.csv --config config.json --output runs/default
```

## Training and inference

```bash
python -m haa_pr fit --config config.json --output runs/default
python -m haa_pr run --config config.json --output runs/default
```

`config.json` retains seed 13 and window/candidate/calibration settings.

## Evaluation

```bash
python -m haa_pr evaluate --config config.json --output runs/default
```

## Comparisons and ablations

`run` computes `full`, `single`, `no-motion`, `no-temporal` and `strongest-peak` controls together.

## Outputs

`windows.json` retains raw candidates and features. `scorer.pt` and `calibration.json` store the pairwise ranker and held-out calibration. `predictions.csv` reports estimates, accepted status, interval unions, inclusion/coverage, truncation and work. Subject-level `metrics.csv` contains MAE/RMSE, accepted MAE/fraction, joint accepted-large-error rate, longest unavailability, harmonic switching, wrong-branch duration, signal recovery, inclusion and cost. `availability_curve.csv` supports matched-availability comparisons.

```bash
python -m haa_pr plot --config config.json --output runs/default
```
