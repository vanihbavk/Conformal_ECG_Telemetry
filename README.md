# Patient-Cluster Conformal Risk Control for Ambulatory ECG Telemetry

[![CI](https://github.com/vanihbavk/Conformal_ECG_Telemetry/actions/workflows/ci.yml/badge.svg)](https://github.com/vanihbavk/Conformal_ECG_Telemetry/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)

Code for the paper **"Distribution-Free Quality Gating for Ambulatory ECG Telemetry via Patient-Cluster Conformal Risk Control"**.
Paper (Zenodo): <https://doi.org/10.5281/zenodo.24312548>

The method wraps any signal-quality score in conformal risk control calibrated over *patients* instead of windows, so the expected rate of silently transmitted clinically failed strips on unseen patients stays below a chosen level `alpha`. The included score `S_0` needs no training labels: five fixed-scale signal-quality features, so every training patient can be used for calibration.

## Results at a glance

Held-out results at `alpha = 0.08` on MIT-BIH with NSTDB noise (47 patients, 20 runs; 95% cluster-bootstrap CI):

| Gate | Accepted strips | Silent clinical risk |
|---|---|---|
| Patient-Cluster CRC with `S_0` | 54.0% | 5.83% [3.53, 8.89] |
| Orphanidou (2015) rule, uncalibrated | 71.1% | 12.07% |

Orphanidou's rule accepts only 43.1% of clean ectopic strips. On CinC 2017 (`alpha = 0.015`), CRC keeps 95.3% of AFib recordings; the Orphanidou rule keeps 47.9%. All numbers are in [`results/`](results/) and are checked by `make test`.

## Repository layout

```text
src/ecgrisk/     library: front end, quality features, conformal calibration, metrics, loaders
experiments/     scripts that produce everything in results/
tests/           unit tests and consistency checks on results/
results/         committed outputs: JSON summaries and figures/
data/            PhysioNet data, created by the download script (not in git)
```

| Script | Writes |
|---|---|
| `experiments/run_benchmark.py` | `results/summary.json`, `results/frozen_score_params.json` |
| `experiments/run_unseen_noise_eval.py` | `results/unseen_noise_results.json` |
| `experiments/run_cinc2011_eval.py` | `results/cinc2011_summary.json` |
| `experiments/run_cinc2017_eval.py` | `results/cinc2017_summary.json` |
| `experiments/draw_all_figures.py` | `results/figures/*.pdf` |

The benchmark and evaluation scripts also write LaTeX table fragments to `results/latex/` (not committed).

## Setup

Python 3.10+.

```bash
git clone https://github.com/vanihbavk/Conformal_ECG_Telemetry.git
cd Conformal_ECG_Telemetry
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
make test
```

`make test` needs no data. It checks the committed results against each other and against the library configuration.

## Data

The datasets are public on PhysioNet and are not stored in this repository.

| Dataset | Used for | Size on disk |
|---|---|---|
| [MIT-BIH Arrhythmia](https://doi.org/10.13026/C2F305) (48 records) | primary benchmark | 90 MB |
| [NSTDB](https://doi.org/10.13026/C2HS3T) (`bw`, `em`, `ma`) | noise injection | 6 MB |
| [CinC 2011](https://physionet.org/content/challenge-2011/1.0.0/) Set A (998 records) | external validation, mobile phone | 115 MB |
| [CinC 2017](https://physionet.org/content/challenge-2017/1.0.0/) training set (8,528 records) | external validation, AliveCor | 160 MB |
| **Total** | | **about 370 MB** |

The CinC 2017 archive (95 MB) is deleted after extraction, so peak disk use is about 465 MB.

```bash
make download-data     # fetch and verify everything into data/
make verify-data       # check existing files, download nothing
```

For a subset, or a different location:

```bash
python3 experiments/download_data.py --datasets mitdb nstdb
python3 experiments/download_data.py --data-dir /path/to/data
```

Downloads go through a temporary `.part` file, so an interrupted run does not leave truncated files. Re-running skips files that are already complete. Other flags: `--force`, `--keep-zip`, `--workers`.

## Reproducing the results

```bash
make benchmark      # MIT-BIH + NSTDB; also writes results/frozen_score_params.json
make unseen-noise   # train on bw+ma noise, test on em; needs frozen_score_params.json
make cinc2011       # needs frozen_score_params.json
make cinc2017       # needs frozen_score_params.json
make figures
```

`make reproduce` runs the download and all of the above in order. The three external evaluations apply the frozen `S_0` parameters from the benchmark, so run `make benchmark` first if you regenerate from scratch. The committed `results/frozen_score_params.json` is enough to run them directly.

## Using the library

```python
from ecgrisk import (
    apply_diagnostic_afe_360hz, causal_clinical_bandpass_360hz,
    extract_sqi_features, calibrate_crc_threshold,
)
```

`extract_sqi_features` returns the five quality features for a 10 s strip at 360 Hz. `calibrate_crc_threshold` selects the threshold from per-patient calibration losses. See `experiments/run_cinc2011_eval.py` for a short end-to-end example on a new cohort.

## License

MIT. See [LICENSE](LICENSE).
