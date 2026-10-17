"""Render + interaction smoke test for the Streamlit app.

Runs every module and exercises the key filters so that layout changes and
widget wiring are checked for exceptions, not just for looks.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = [
    "1 · Overview",
    "2 · Individual monitoring",
    "3 · Forecasting",
    "4 · Anomaly review",
    "5 · Forecast evaluation",
]


def fresh():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, f"initial run: {at.exception}"
    return at


def goto(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
    return at


for page in PAGES:
    at = goto(fresh(), page)
    print(f"OK: {page}")

# Module 2: every participant, first analyte
at = goto(fresh(), "2 · Individual monitoring")
for subject in at.selectbox[0].options:
    at.selectbox[0].set_value(subject).run()
    assert not at.exception, f"Module 2 subject {subject}: {at.exception}"
print("OK: Module 2 participants")

# Module 3: horizon extremes
at = goto(fresh(), "3 · Forecasting")
for h in (1, 3):
    at.slider[0].set_value(h).run()
    assert not at.exception, f"Module 3 horizon {h}: {at.exception}"
print("OK: Module 3 horizons")

# Module 4: every threshold x participant filter
at = goto(fresh(), "4 · Anomaly review")
for th in at.selectbox[3].options:
    at.selectbox[3].set_value(th).run()
    assert not at.exception, f"Module 4 threshold {th}: {at.exception}"
    for subject in at.selectbox[0].options:
        at.selectbox[0].set_value(subject).run()
        assert not at.exception, f"Module 4 {th}/{subject}: {at.exception}"
print("OK: Module 4 thresholds x participants")

# Module 4: panel + timepoint filters
at = goto(fresh(), "4 · Anomaly review")
for p in at.selectbox[1].options:
    at.selectbox[1].set_value(p).run()
    assert not at.exception, f"Module 4 panel {p}: {at.exception}"
for t in at.selectbox[2].options:
    at.selectbox[2].set_value(t).run()
    assert not at.exception, f"Module 4 timepoint {t}: {at.exception}"
print("OK: Module 4 panels x timepoints")

print("\nall modules and filters render without exceptions")
