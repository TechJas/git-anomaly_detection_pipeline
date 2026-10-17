import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path("data/processed")
DASH = PROC / "astrohealth_dashboard.csv"

cbc_cleaned = pd.read_csv(PROC / "cbc_cleaned.csv")
cmp_cleaned = pd.read_csv(PROC / "cmp_cleaned.csv")

# --- CBC: already in long format ---
cbc_cols = ["SUBJECT_ID", "TEST_DATE", "ANALYTE", "VALUE", "UNITS", "RANGE_MIN", "RANGE_MAX"]
cbc = cbc_cleaned[cbc_cols].copy()
cbc["PANEL"] = "CBC"

# --- CMP: wide -> long, one block per value column ---
rows = []
for col in cmp_cleaned.columns:
    if "_value_" not in col and not col.endswith("_value"):
        continue
    if "_value_" in col:
        name, unit = col.split("_value_", 1)
        analyte = col.replace("_value_", "_", 1)
    else:
        name = col.replace("_value", "")
        unit = ""
        analyte = name
    minc = f"{name}_range_min_{unit}" if unit else f"{name}_range_min"
    maxc = f"{name}_range_max_{unit}" if unit else f"{name}_range_max"
    part = cmp_cleaned[["SUBJECT_ID", "TEST_DATE", col, minc, maxc]].copy()
    part.columns = ["SUBJECT_ID", "TEST_DATE", "VALUE", "RANGE_MIN", "RANGE_MAX"]
    part["ANALYTE"] = analyte
    part["UNITS"] = unit if unit else pd.NA
    part["PANEL"] = "CMP"
    rows.append(part)
cmp_long = pd.concat(rows, ignore_index=True)

# --- Combine ---
cols = ["SUBJECT_ID", "TEST_DATE", "PANEL", "ANALYTE", "VALUE", "UNITS", "RANGE_MIN", "RANGE_MAX"]
data = pd.concat([cbc, cmp_long], ignore_index=True)[cols]

# --- Reference-range status ---
cond = [
    data["VALUE"].isna() | data["RANGE_MIN"].isna() | data["RANGE_MAX"].isna(),
    data["VALUE"] < data["RANGE_MIN"],
    data["VALUE"] > data["RANGE_MAX"],
]
data["range_status"] = np.select(cond, ["missing", "below", "above"], default="within")

# --- Personal baselines from preflight ---
PRE = ["L-92", "L-44", "L-3"]
POST = ["R+1", "R+45", "R+82", "R+194"]

pre = data[data["TEST_DATE"].isin(PRE)]
base = (
    pre.groupby(["SUBJECT_ID", "PANEL", "ANALYTE"])["VALUE"]
    .agg(base_count="count", base_mean="mean", base_std="std")
    .reset_index()
)

# --- Score postflight against baseline ---
post = data[data["TEST_DATE"].isin(POST)].copy()
scores = post.merge(base, on=["SUBJECT_ID", "PANEL", "ANALYTE"], how="left")
scores["z_score"] = (scores["VALUE"] - scores["base_mean"]) / scores["base_std"]

# --- Multi-threshold flags ---
for th, c in [(2.0, "flag_2_0"), (2.5, "flag_2_5"), (3.0, "flag_3_0")]:
    scores[c] = (scores["z_score"].abs() > th).astype("boolean")
    scores.loc[scores["z_score"].isna(), c] = pd.NA

# --- Clear scoring rules ---
scorable = scores["z_score"].notna()
scores["baseline_status"] = "unscored"
scores.loc[scorable, "baseline_status"] = "within_baseline_threshold"
scores.loc[scorable & (scores["z_score"].abs() > 2.0), "baseline_status"] = "baseline_deviation"

# --- Export ---
out_cols = cols + [
    "range_status", "base_mean", "base_std", "base_count",
    "z_score", "baseline_status", "flag_2_0", "flag_2_5", "flag_3_0",
]
out = scores[out_cols].copy()
out.to_csv(DASH, index=False)

# --- Full panel (all timepoints) with baseline context for trend charts ---
panel = data.merge(base, on=["SUBJECT_ID", "PANEL", "ANALYTE"], how="left")
panel["z_score"] = (panel["VALUE"] - panel["base_mean"]) / panel["base_std"]
panel.to_csv(PROC / "astrohealth_panel.csv", index=False)

print("Rows:", len(out))
print("baseline_status:", out["baseline_status"].value_counts().to_dict())
print("flags:", int(out["flag_2_0"].sum()), int(out["flag_2_5"].sum()), int(out["flag_3_0"].sum()))
print("columns:", list(out.columns))
