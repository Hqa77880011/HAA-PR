# HAA-PR

Implementation of **Harmonic-ambiguity-aware pulse-rate measurement under motion.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

Use synchronized PPG/reference data such as [BIDMC](https://physionet.org/content/bidmc/1.0.0/) and the paper's exercise recordings. Input CSV columns are `time,ppg,reference,subject,record,split`; optional `ax,ay,az` add motion. Roles are `train,cal,test`, disjoint by subject. Resample each record to `sample_rate` before import. Reference rate is aligned to the same trailing analysis window. The CSV reference is an ECG-derived operational target; PPG-derived labels are not an independent reference.

## Workflow

```bash
python -m haa_pr prepare --input data/ppg.csv
python -m haa_pr fit
python -m haa_pr run
python -m haa_pr evaluate
python -m haa_pr plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`window_seconds,stride_seconds` define causal windows. `minimum_rate,maximum_rate,prominence,merge_bpm` define the candidate generator. `horizon` recomputes finite-window path recursion from scratch. `transition_weight,transition_scale` bound temporal preference. `margin_grid,retention_target,localization_quantile` calibrate retention/width. Clock, alignment, slope and reference bounds widen the union; `acceptance_bpm` is its reporting radius limit. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`windows.json` retains raw candidates and features. `scorer.pt` and `calibration.json` store the pairwise ranker and held-out calibration. `predictions.csv` reports estimates, accepted status, interval unions, inclusion/coverage, truncation and work. Subject-level `metrics.csv` contains MAE/RMSE, accepted MAE/fraction, joint accepted-large-error rate, longest unavailability, harmonic switching, wrong-branch duration, signal recovery, inclusion and cost. `availability_curve.csv` supports matched-availability comparisons.

## Implementation and controls

The two-hidden-layer scorer uses spectral, harmonic, autocorrelation and motion evidence. A bounded transition penalty and finite-horizon reset avoid indefinitely accumulating an old branch preference. Multiple endpoint intervals remain a union. Acceptance uses radius about a supported candidate plus validity/truncation flags. FFT padding interpolates the grid; calibration, rather than padding density, determines localization width.

Controls: `full`, `single`, `no-motion`, `no-temporal`, `strongest-peak`. The run command evaluates the listed controls together.

```bash
python -m haa_pr.compare --inputs "runs/*/metrics.csv" --reference full --metrics mae accepted_mae accepted_fraction --unit subject
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

