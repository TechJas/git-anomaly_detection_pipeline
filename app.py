"""NASA Inspiration4 Astronaut Health Monitoring and Biomarker Forecasting Dashboard.

An explainable research prototype that (1) monitors individual biomarker changes
against personal preflight baselines and (2) evaluates whether historical CBC and
CMP measurements can support short-term forecasting.

This tool does not diagnose disease and does not predict clinical deterioration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PROC = BASE_DIR / "data" / "processed"

TIMEPOINTS = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82", "R+194"]
FLAG_COLS = {"2.0": "flag_2_0", "2.5": "flag_2_5", "3.0": "flag_3_0"}
MODEL_LABELS = {
    "naive_last": "Last value (naive)",
    "mean_history": "History mean",
    "linear_trend": "Linear trend",
}
MODEL_COLORS = {
    "naive_last": "#636EFA",
    "mean_history": "#00CC96",
    "linear_trend": "#EF553B",
}
THRESHOLD_NOTES = {
    "2.0": "Flags any measurement more than **2 standard deviations** from the participant's own "
    "preflight baseline. Under a normal distribution about 5% of values exceed this by chance.",
    "2.5": "Flags any measurement more than **2.5 standard deviations** from the participant's own "
    "preflight baseline. About 1% of values exceed this by chance.",
    "3.0": "Flags any measurement more than **3 standard deviations** from the participant's own "
    "preflight baseline. About 0.3% of values exceed this by chance.",
}

st.set_page_config(
    page_title="Astronaut Health & Biomarker Dashboard",
    page_icon="🛰️",
    layout="wide",
)


@st.cache_data
def load_tables():
    dash = pd.read_csv(PROC / "astrohealth_dashboard.csv")
    panel = pd.read_csv(PROC / "astrohealth_panel.csv")
    panel["TEST_DATE"] = pd.Categorical(panel["TEST_DATE"], categories=TIMEPOINTS, ordered=True)
    panel = panel.sort_values(["SUBJECT_ID", "ANALYTE", "TEST_DATE"])
    overall = pd.read_csv(PROC / "forecast_overall.csv")
    by_step = pd.read_csv(PROC / "forecast_by_step.csv")
    return dash, panel, overall, by_step


def fit_and_forecast(history: pd.DataFrame, horizon: int):
    """Fit simple models on a history and forecast `horizon` steps ahead."""
    y = history["VALUE"].to_numpy(dtype=float)
    x = np.arange(len(y))
    preds = {}

    preds["naive_last"] = np.repeat(y[-1], horizon)
    preds["mean_history"] = np.repeat(y.mean(), horizon)

    if len(y) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        fx = np.arange(len(y), len(y) + horizon)
        preds["linear_trend"] = slope * fx + intercept
    else:
        preds["linear_trend"] = np.repeat(y[-1], horizon)

    return preds


try:
    dash, panel, overall, by_step = load_tables()
except FileNotFoundError as exc:
    st.error(
        f"Missing processed data file: {exc.filename}. "
        "Run `python scripts/build_dashboard.py` and "
        "`python scripts/forecast_eval.py` first."
    )
    st.stop()

st.sidebar.title("🛰️ AstroHealth")
page = st.sidebar.radio(
    "Module",
    [
        "1 · Overview",
        "2 · Individual monitoring",
        "3 · Forecasting",
        "4 · Anomaly review",
        "5 · Forecast evaluation",
    ],
)
st.sidebar.divider()
st.sidebar.caption("Research prototype · not for clinical use")
st.sidebar.caption("Data: NASA LSDS · Inspiration4 crew (C001–C004)")


# ---------------------------------------------------------------- Module 1
if page.startswith("1"):
    st.title("Health Data Overview")
    st.caption("CBC and CMP measurements across mission timepoints for the Inspiration4 crew.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Participants", panel["SUBJECT_ID"].nunique(), help="Crew members C001–C004")
    c2.metric("Lab panels", panel["PANEL"].nunique(), help="CBC = Complete Blood Count, CMP = Comprehensive Metabolic Panel")
    c3.metric("Analytes", panel["ANALYTE"].nunique(), help="Distinct measured biomarkers")
    c4.metric(
        "Scored measurements",
        int(dash["z_score"].notna().sum()),
        help="Postflight measurements with a computable personal-baseline z-score",
    )

    st.divider()
    st.subheader("Measurements by panel and timepoint")
    counts = (
        panel.groupby(["TEST_DATE", "PANEL"], observed=True)
        .size()
        .reset_index(name="measurements")
    )
    fig = px.bar(
        counts,
        x="TEST_DATE",
        y="measurements",
        color="PANEL",
        barmode="group",
        labels={"TEST_DATE": "Mission timepoint (L- = preflight, R+ = postflight)", "measurements": "Number of measurements"},
    )
    fig.update_layout(legend_title_text="Panel", xaxis_tickangle=0)
    st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Data quality")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Missing values per column**")
        st.dataframe(
            panel.isna().sum().rename("missing_count").to_frame(),
            width="stretch",
        )
    with q2:
        st.markdown("**Baseline status** (scored postflight measurements)")
        st.dataframe(
            dash["baseline_status"].value_counts().rename("count").to_frame(),
            width="stretch",
        )
        st.markdown("**Reference-range status** (vs population limits)")
        st.dataframe(
            dash["range_status"].value_counts().rename("count").to_frame(),
            width="stretch",
        )


# ---------------------------------------------------------------- Module 2
elif page.startswith("2"):
    st.title("Individual Biomarker Monitoring")
    st.caption(
        "Historical values for one participant, compared with their own preflight baseline "
        "and the population reference range."
    )

    subjects = sorted(panel["SUBJECT_ID"].unique())
    col1, col2 = st.columns(2)
    subject = col1.selectbox("Participant", subjects)
    subj_panel = panel[panel["SUBJECT_ID"] == subject]
    analyte = col2.selectbox("Analyte", sorted(subj_panel["ANALYTE"].unique()))

    series = subj_panel[subj_panel["ANALYTE"] == analyte].sort_values("TEST_DATE")
    row = series.iloc[0]
    base_mean, base_std = row["base_mean"], row["base_std"]
    rmin, rmax, units = row["RANGE_MIN"], row["RANGE_MAX"], row["UNITS"]
    unit_label = units if isinstance(units, str) and units else "value"

    latest = series.dropna(subset=["VALUE"]).iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest value", f"{latest['VALUE']:.1f} {unit_label}")
    m2.metric("Personal baseline mean", f"{base_mean:.1f}" if pd.notna(base_mean) else "n/a")
    m3.metric("Personal baseline SD", f"{base_std:.2f}" if pd.notna(base_std) else "n/a")
    m4.metric(
        "Reference range",
        f"{rmin:.0f}–{rmax:.0f}" if pd.notna(rmin) and pd.notna(rmax) else "n/a",
    )

    st.divider()
    fig = go.Figure()
    if pd.notna(base_std):
        xs = list(series["TEST_DATE"].astype(str))
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=[base_mean + base_std] * len(xs) + [base_mean - base_std] * len(xs),
                fill="toself",
                fillcolor="rgba(120,120,120,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Personal baseline ±1 SD",
                hoverinfo="skip",
            )
        )
        fig.add_hline(
            y=base_mean, line_dash="dash", line_color="grey",
            annotation_text="Baseline mean", annotation_position="right",
        )
    if pd.notna(rmin) and pd.notna(rmax):
        fig.add_hline(
            y=rmax, line_dash="dot", line_color="green",
            annotation_text="Reference max", annotation_position="right",
        )
        fig.add_hline(
            y=rmin, line_dash="dot", line_color="green",
            annotation_text="Reference min", annotation_position="right",
        )
    fig.add_trace(
        go.Scatter(
            x=series["TEST_DATE"].astype(str),
            y=series["VALUE"],
            mode="lines+markers",
            name="Observed",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title=f"{analyte} — {subject}",
        xaxis_title="Mission timepoint",
        yaxis_title=unit_label,
        legend_title_text="Series",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The grey band is ±1 standard deviation around the participant's own preflight mean. "
        "Green dotted lines are the population reference range, which is a different comparison."
    )

    st.divider()
    st.subheader("Postflight measurements and personal-baseline z-scores")
    post_view = dash[(dash["SUBJECT_ID"] == subject) & (dash["ANALYTE"] == analyte)][
        ["TEST_DATE", "VALUE", "base_mean", "base_std", "z_score", "range_status", "baseline_status"]
    ].rename(
        columns={
            "TEST_DATE": "Timepoint",
            "VALUE": "Value",
            "base_mean": "Baseline mean",
            "base_std": "Baseline SD",
            "z_score": "z-score",
            "range_status": "Range status",
            "baseline_status": "Baseline status",
        }
    )
    st.dataframe(post_view, hide_index=True, width="stretch")


# ---------------------------------------------------------------- Module 3
elif page.startswith("3"):
    st.title("Future Biomarker Forecasting")
    st.caption(
        "Estimate future values from a participant's history using simple models. "
        "With only 7 timepoints per analyte these forecasts are illustrative, not clinical."
    )

    subjects = sorted(panel["SUBJECT_ID"].unique())
    col1, col2, col3 = st.columns(3)
    subject = col1.selectbox("Participant", subjects)
    subj_panel = panel[panel["SUBJECT_ID"] == subject]
    analyte = col2.selectbox("Analyte", sorted(subj_panel["ANALYTE"].unique()))
    horizon = col3.slider("Forecast horizon (timepoints)", 1, 3, 1)

    series = subj_panel[subj_panel["ANALYTE"] == analyte].dropna(subset=["VALUE"]).sort_values("TEST_DATE")
    if len(series) < 3:
        st.warning("Not enough history to forecast this analyte.")
    else:
        units = series.iloc[0]["UNITS"]
        unit_label = units if isinstance(units, str) and units else "value"
        preds = fit_and_forecast(series, horizon)
        future_x = [f"F+{i + 1}" for i in range(horizon)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series["TEST_DATE"].astype(str),
                y=series["VALUE"],
                mode="lines+markers",
                name="Observed (measured)",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
            )
        )
        for model, values in preds.items():
            fig.add_trace(
                go.Scatter(
                    x=[series["TEST_DATE"].astype(str).iloc[-1]] + future_x,
                    y=[series["VALUE"].iloc[-1]] + list(values),
                    mode="lines+markers",
                    name=f"{MODEL_LABELS[model]} (forecast)",
                    line=dict(dash="dot", color=MODEL_COLORS[model]),
                    marker=dict(symbol="diamond"),
                )
            )
        fig.add_vline(
            x=len(series) - 0.5, line_dash="dot", line_color="grey",
            annotation_text="forecast →", annotation_position="top",
        )
        fig.update_layout(
            title=f"{analyte} — {subject}",
            xaxis_title="Mission timepoint (F+ = forecast, not measured)",
            yaxis_title=unit_label,
            legend_title_text="Series",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        st.info(
            "The F+ points on the right are **estimates**, not measurements. "
            "No prediction intervals are reported because this short series cannot support a "
            "reliable error estimate for an individual forecast."
        )

        st.subheader("Model definitions")
        st.markdown(
            "- **Last value (naive):** repeats the most recent measurement.\n"
            "- **History mean:** repeats the average of all past measurements.\n"
            "- **Linear trend:** extends a straight line fitted to the history."
        )

        forecast_table = pd.DataFrame(
            {MODEL_LABELS[m]: v for m, v in preds.items()},
            index=pd.Index(future_x, name="Forecast timepoint"),
        )
        st.dataframe(forecast_table.round(2), width="stretch")
        st.caption(f"Estimated values in {unit_label}.")


# ---------------------------------------------------------------- Module 4
elif page.startswith("4"):
    st.title("Anomaly Review")
    st.caption("Measurements that deviate from a participant's own preflight baseline.")

    col1, col2, col3, col4 = st.columns(4)
    subjects = ["All"] + sorted(dash["SUBJECT_ID"].unique())
    panels = ["All"] + sorted(dash["PANEL"].unique())
    timepoints = ["All"] + [t for t in TIMEPOINTS if t in set(dash["TEST_DATE"])]
    subject = col1.selectbox("Participant", subjects)
    panel_sel = col2.selectbox("Panel", panels)
    timepoint = col3.selectbox("Timepoint", timepoints)
    threshold = col4.selectbox("Threshold (|z|)", list(FLAG_COLS), index=0)

    with st.expander("What does a flag mean?"):
        st.markdown(
            "A flag compares a measurement with the **same participant's own preflight baseline**:"
        )
        st.latex(r"z = \frac{\text{value} - \text{personal baseline mean}}{\text{personal baseline SD}}")
        st.markdown(THRESHOLD_NOTES[threshold])
        st.markdown(
            "A flag means the value is **unusual relative to that person's normal range** — it is "
            "not a diagnosis. The separate `Range status` column compares the value with the "
            "population reference range and is a different check."
        )

    st.info(THRESHOLD_NOTES[threshold])

    view = dash.copy()
    if subject != "All":
        view = view[view["SUBJECT_ID"] == subject]
    if panel_sel != "All":
        view = view[view["PANEL"] == panel_sel]
    if timepoint != "All":
        view = view[view["TEST_DATE"] == timepoint]

    flag_col = FLAG_COLS[threshold]
    flagged = view[view[flag_col] == True].copy()
    flagged["abs_z"] = flagged["z_score"].abs()

    k1, k2, k3 = st.columns(3)
    k1.metric("Flagged measurements", len(flagged))
    k2.metric("Participant(s) affected", flagged["SUBJECT_ID"].nunique())
    k3.metric("Analytes affected", flagged["ANALYTE"].nunique())

    st.divider()
    st.subheader("Flagged measurements")
    flagged_view = flagged[
        ["SUBJECT_ID", "TEST_DATE", "PANEL", "ANALYTE", "VALUE", "UNITS", "base_mean", "base_std", "z_score", "range_status"]
    ].rename(
        columns={
            "SUBJECT_ID": "Participant",
            "TEST_DATE": "Timepoint",
            "PANEL": "Panel",
            "ANALYTE": "Analyte",
            "VALUE": "Value",
            "UNITS": "Units",
            "base_mean": "Baseline mean",
            "base_std": "Baseline SD",
            "z_score": "z-score",
            "range_status": "Range status",
        }
    ).sort_values("z-score", key=lambda s: s.abs(), ascending=False)
    st.dataframe(
        flagged_view,
        hide_index=True,
        width="stretch",
        column_config={
            "Value": st.column_config.NumberColumn(format="%.2f"),
            "Baseline mean": st.column_config.NumberColumn(format="%.2f"),
            "Baseline SD": st.column_config.NumberColumn(format="%.2f"),
            "z-score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.divider()
    st.subheader("Flag counts across all thresholds")
    st.dataframe(
        pd.DataFrame(
            {
                "Threshold (|z|)": list(FLAG_COLS),
                "Flagged measurements": [int(dash[c].sum()) for c in FLAG_COLS.values()],
            }
        ),
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------------- Module 5
else:
    st.title("Forecast Evaluation")
    st.caption(
        "Chronological (walk-forward) one-step-ahead validation. "
        "Each timepoint is predicted using only earlier measurements."
    )

    st.subheader("Overall model comparison")
    table = overall.copy()
    table["Model"] = table["MODEL"].map(MODEL_LABELS)
    table["Median error vs naive (MAE)"] = table["median_relMAE"]
    table["Median error vs naive (RMSE)"] = table["median_relRMSE"]
    table["Beats naive (MAE)"] = table["win_rate_MAE"] * 100
    table = table[["Model", "Median error vs naive (MAE)", "Median error vs naive (RMSE)", "Beats naive (MAE)", "N_series"]].rename(
        columns={"N_series": "Series"}
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Median error vs naive (MAE)": st.column_config.NumberColumn("Median error vs naive (MAE)", format="%.2f"),
            "Median error vs naive (RMSE)": st.column_config.NumberColumn("Median error vs naive (RMSE)", format="%.2f"),
            "Beats naive (MAE)": st.column_config.NumberColumn("Beats naive (MAE)", format="%.0f%%"),
            "Series": st.column_config.NumberColumn("Series", format="%d"),
        },
    )
    st.caption(
        "Errors are compared **within each series** against the last-value baseline, so analytes "
        "with different units are never mixed. The table reports the median across series, which "
        "is robust to a few explosive predictions. A value of 1.00 means the model ties the "
        "baseline; below 1.00 is better."
    )

    best = overall.sort_values("median_relRMSE").iloc[0]
    st.success(
        f"Best model by median relative RMSE: **{MODEL_LABELS[best['MODEL']]}** "
        f"({best['median_relRMSE']:.2f}× the baseline error, better than naive in "
        f"{best['win_rate_RMSE'] * 100:.0f}% of series). The gain over a simple last-value "
        "forecast is small, which is expected with so few timepoints."
    )
    st.warning(
        "Linear trend is the worst model: with only 3–7 points it extrapolates noise and "
        "overshoots. More complex models are not justified on this data."
    )

    st.divider()
    st.subheader("Error by mission timepoint")
    step = by_step.copy()
    step["Model"] = step["MODEL"].map(MODEL_LABELS)
    fig = px.bar(
        step,
        x="STEP",
        y="median_scaled_AE",
        color="Model",
        barmode="group",
        labels={"STEP": "Mission timepoint", "median_scaled_AE": "Median error ÷ naive error"},
    )
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="grey",
        annotation_text="ties naive baseline", annotation_position="top left",
    )
    fig.update_layout(legend_title_text="Model")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Median prediction error scaled by the last-value baseline's error (1.0 = ties naive, "
        "lower is better). The first postflight timepoint (R+1) is hardest because the preflight "
        "baseline is stable while the observed value jumps."
    )

    st.divider()
    st.subheader("Limitations")
    st.markdown(
        "- Only 7 timepoints per participant/analyte; the walk-forward window is very short.\n"
        "- No outcome labels, so this is **not** prediction of disease or clinical deterioration.\n"
        "- Linear trend is unstable and overshoots; more complex models are not justified here.\n"
        "- Forecasting is a feasibility demonstration, not a validated clinical tool."
    )
