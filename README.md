# AstroHealth — NASA Inspiration4 Astronaut Health Monitoring & Biomarker Forecasting Dashboard

An explainable research prototype that monitors individual astronaut biomarker changes
against their own preflight baselines, and tests whether short-term biomarker values can
be forecast from history.

**Live demo:** (https://techjas-git-anomaly-detection-pipeline-app-a5atpx.streamlit.app/)

> ⚠️ Research prototype. This tool does not diagnose disease and does not predict clinical
> deterioration. It is a data-science portfolio project, not a medical device.

---

## The Problem

After spaceflight, astronauts show measurable shifts in blood biomarkers. Standard lab
reports compare each value to a **population reference range**, which has two weaknesses:

1. It ignores that a value can be *normal* for the population but *unusual for a specific person*.
2. It gives no sense of whether a trend is likely to continue or reverse.

Clinicians increasingly favour **personal-baseline (serial) monitoring** — tracking a patient
against their own history. I wanted to build an explainable tool that does this for the
Inspiration4 crew's CBC and CMP data, and to honestly test whether the data can support
short-term forecasting.

## The Solution

A Streamlit dashboard with five modules that:

- reports data quality and coverage,
- tracks each biomarker against a **personal preflight baseline**,
- flags measurements that deviate from that baseline (z-score based),
- produces **short-term forecasts** with three simple models, and
- **evaluates** those forecasts against a last-value baseline using chronological validation.

## Dataset

| Item | Value |
|------|-------|
| Source | NASA Life Sciences Data Archive (LSDS) — Inspiration4 |
| Participants | 4 crew members (C001–C004) |
| Panels | CBC (20 analytes) + CMP (16 analytes) |
| Analytes | 36 |
| Timepoints | 7 — preflight `L-92, L-44, L-3` and postflight `R+1, R+45, R+82, R+194` |
| Measurements | 1,001 |

## Approach

```
raw CBC / CMP CSV
      │  notebooks/cbc_eda_and_cleaning.ipynb
      │  notebooks/cmp_eda_and_cleaning.ipynb
      ▼
cleaned long-format data (cbc_cleaned.csv, cmp_cleaned.csv)
      │  scripts/build_dashboard.py
      │   ├─ personal preflight baselines (mean, SD, count)
      │   ├─ per-measurement z-score vs baseline
      │   ├─ multi-threshold flags (|z| > 2.0 / 2.5 / 3.0)
      │   └─ reference-range status
      ▼
dashboard + panel datasets
      │  scripts/forecast_eval.py
      │   └─ chronological walk-forward evaluation of 3 models
      ▼
Streamlit dashboard (app.py)
```

**Personal-baseline anomaly score**

```
z = (value − personal preflight mean) / personal preflight SD
```

A measurement is flagged when `|z|` exceeds a chosen threshold. This is an **exploratory
signal**, not a diagnosis: it says "unusual for this person", nothing more.

## Results

### 1. Personal-baseline anomaly detection (threshold sensitivity)

| Threshold \|z\| | Flagged measurements |
|-----------------|----------------------|
| 2.0 | 132 |
| 2.5 | 93 |
| 3.0 | 79 |

Reference-range status of scored postflight measurements: 555 within range, 10 above,
8 below, 1 missing.

### 2. Forecast evaluation (walk-forward one-step-ahead)

Models are compared **within each series** against a last-value (naive) baseline, and the
summary uses the **median** so a few explosive predictions don't distort the picture.

| Model | Median error vs naive (MAE) | Median error vs naive (RMSE) | Beats naive |
|-------|-----------------------------|------------------------------|-------------|
| History mean | **0.83** | **0.80** | **72%** |
| Linear trend | 1.16 | 1.18 | 19% |
| Last value (naive) | 1.00 | 1.00 | — |

**Finding:** a simple history mean slightly outperforms the last-value baseline, while a
linear trend is the worst model (it extrapolates noise from only a handful of points). The
gain over naive is small — which is the honest expectation with 7 timepoints per analyte.

## Dashboard Modules

1. **Overview** — participants, panels, analytes, missingness, flag counts.
2. **Individual monitoring** — pick a participant + analyte; see history, baseline ±1 SD band,
   and population reference range.
3. **Forecasting** — project the next 1–3 timepoints with three models (clearly marked as estimates).
4. **Anomaly review** — filter flagged measurements by participant, panel, timepoint and threshold,
   with a plain-language explanation of what a flag means.
5. **Forecast evaluation** — robust model comparison, error by timepoint, and limitations.

## Tech Stack

`Python` · `pandas` · `NumPy` · `Plotly` · `Streamlit`

## Project Structure

```
.
├── app.py                     # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                   # original NASA files (not tracked)
│   └── processed/             # cleaned + generated datasets (tracked)
├── notebooks/
│   ├── cbc_eda_and_cleaning.ipynb
│   ├── cmp_eda_and_cleaning.ipynb
│   └── unified_biomarker_pipeline.ipynb
└── scripts/
    ├── build_dashboard.py     # baselines, z-scores, flags
    ├── forecast_eval.py       # chronological forecast evaluation
    └── smoke_test.py          # render/interaction tests
```

## Run Locally

```bash
pip install -r requirements.txt

# regenerate the processed datasets (optional — already committed)
python scripts/build_dashboard.py
python scripts/forecast_eval.py

# launch the dashboard
streamlit run app.py
```

Run the tests:

```bash
python scripts/smoke_test.py
```

## Limitations

- Only **4 participants** and **7 timepoints** per analyte — no statistical power for a risk model.
- **No outcome labels**, so this is not prediction of disease or clinical deterioration.
- Forecasting is a **feasibility demonstration**; no prediction intervals are reported because
  the short series cannot support a reliable per-forecast error estimate.
- Personal baselines use only 3 preflight points, so the baseline SD is itself uncertain.
- Numeric values retain the units of the source files; normalization is applied for comparison.

## Future Work

- Larger longitudinal cohorts with outcome labels.
- Reference-change-value (RCV) thresholds instead of fixed z cut-offs.
- Calibrated prediction intervals once series are long enough.
- External validation before any clinical claim.

## Acknowledgements

Data: NASA Life Sciences Data Archive (LSDS) — Inspiration4 mission.
Built as a personal learning project in health data analytics.
