"""Chronological forecast evaluation.

Walk-forward one-step-ahead validation: each timepoint is predicted using only
earlier measurements. Because analytes have wildly different units and scales,
models are compared *within each series* against the last-value (naive)
baseline, and the summary uses the median (robust to a few explosive series).

Outputs (all in data/processed/):
  forecast_predictions.csv  one row per (series, step, model)
  forecast_eval.csv         per-series MAE/RMSE and error relative to naive
  forecast_overall.csv      per-model robust summary
  forecast_by_step.csv      per-timepoint median scaled error
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROC = Path("data/processed")
TIMEPOINTS = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82", "R+194"]
MODELS = ("naive_last", "mean_history", "linear_trend")


def load_long():
    """Reconstruct the full long panel (pre + post) from the cleaned files."""
    cbc = pd.read_csv(PROC / "cbc_cleaned.csv")
    cbc = cbc[["SUBJECT_ID", "TEST_DATE", "ANALYTE", "VALUE", "UNITS"]].copy()
    cbc["PANEL"] = "CBC"

    cmp = pd.read_csv(PROC / "cmp_cleaned.csv")
    rows = []
    for col in cmp.columns:
        if "_value_" not in col and not col.endswith("_value"):
            continue
        if "_value_" in col:
            name, unit = col.split("_value_", 1)
            analyte = col.replace("_value_", "_", 1)
        else:
            name = col.replace("_value", "")
            unit = ""
            analyte = name
        part = cmp[["SUBJECT_ID", "TEST_DATE", col]].copy()
        part.columns = ["SUBJECT_ID", "TEST_DATE", "VALUE"]
        part["ANALYTE"] = analyte
        part["UNITS"] = unit if unit else pd.NA
        part["PANEL"] = "CMP"
        rows.append(part)
    cmp_long = pd.concat(rows, ignore_index=True)

    data = pd.concat([cbc, cmp_long], ignore_index=True)
    data["TEST_DATE"] = pd.Categorical(data["TEST_DATE"], categories=TIMEPOINTS, ordered=True)
    return data.dropna(subset=["VALUE"]).sort_values("TEST_DATE")


def forecast_one(name, history):
    y = history["VALUE"].to_numpy(dtype=float)
    if name == "naive_last":
        return y[-1]
    if name == "mean_history":
        return y.mean()
    if name == "linear_trend":
        if len(y) < 2:
            return y[-1]
        slope, intercept = np.polyfit(np.arange(len(y)), y, 1)
        return slope * len(y) + intercept
    return np.nan


def walk_forward(data):
    records = []
    for (sid, panel, analyte), grp in data.groupby(["SUBJECT_ID", "PANEL", "ANALYTE"], observed=True):
        grp = grp.sort_values("TEST_DATE")
        if len(grp) < 3:
            continue
        for i in range(2, len(grp)):
            history = grp.iloc[:i]
            target = grp.iloc[i]
            for name in MODELS:
                records.append(
                    {
                        "SUBJECT_ID": sid,
                        "PANEL": panel,
                        "ANALYTE": analyte,
                        "STEP": str(target["TEST_DATE"]),
                        "MODEL": name,
                        "OBSERVED": target["VALUE"],
                        "PREDICTED": forecast_one(name, history),
                    }
                )
    return pd.DataFrame(records)


def build_outputs(preds):
    preds = preds.dropna(subset=["PREDICTED", "OBSERVED"]).copy()
    preds["error"] = preds["PREDICTED"] - preds["OBSERVED"]
    preds["abs_error"] = preds["error"].abs()
    preds["sq_error"] = preds["error"] ** 2

    keys = ["SUBJECT_ID", "PANEL", "ANALYTE"]

    # Per-series MAE/RMSE, then compare each model to naive within the same series.
    per_series = (
        preds.groupby(keys + ["MODEL"])
        .agg(MAE=("abs_error", "mean"), RMSE=("sq_error", lambda s: np.sqrt(s.mean())), N=("error", "size"))
        .reset_index()
    )
    naive = per_series[per_series["MODEL"] == "naive_last"].set_index(keys)[["MAE", "RMSE"]]
    naive = naive.replace(0, np.nan)
    per_series = per_series.join(naive, on=keys, rsuffix="_naive")
    per_series["relMAE"] = per_series["MAE"] / per_series["MAE_naive"]
    per_series["relRMSE"] = per_series["RMSE"] / per_series["RMSE_naive"]

    overall = (
        per_series.groupby("MODEL")
        .apply(
            lambda d: pd.Series(
                {
                    "median_relMAE": d["relMAE"].median(),
                    "median_relRMSE": d["relRMSE"].median(),
                    "win_rate_MAE": (d["relMAE"] < 1).mean(),
                    "win_rate_RMSE": (d["relRMSE"] < 1).mean(),
                    "avg_MAE": d["MAE"].mean(),
                    "avg_RMSE": d["RMSE"].mean(),
                    "N_series": len(d),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # Scale each prediction by the naive MAE of its own series -> unit-free per step.
    naive_mae = per_series[per_series["MODEL"] == "naive_last"].set_index(keys)["MAE"]
    preds = preds.join(naive_mae.rename("naive_MAE"), on=keys)
    preds["scaled_ae"] = preds["abs_error"] / preds["naive_MAE"].replace(0, np.nan)
    by_step = (
        preds.groupby(["STEP", "MODEL"], observed=True)
        .agg(median_scaled_AE=("scaled_ae", "median"), N=("error", "size"))
        .reset_index()
    )
    return per_series, overall, by_step


if __name__ == "__main__":
    data = load_long()
    preds = walk_forward(data)
    preds.to_csv(PROC / "forecast_predictions.csv", index=False)

    per_series, overall, by_step = build_outputs(preds)
    per_series.to_csv(PROC / "forecast_eval.csv", index=False)
    overall.to_csv(PROC / "forecast_overall.csv", index=False)
    by_step.to_csv(PROC / "forecast_by_step.csv", index=False)

    print("Predictions:", len(preds))
    print(overall.to_string(index=False))
