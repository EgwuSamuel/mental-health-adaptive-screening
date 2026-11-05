# Certified Adaptive Screening (CAS) for depression and anxiety

**FUNAAB · Department of Computer Science**

CAS is a short, adaptive version of the PHQ-9 (depression) and GAD-7 (anxiety)
questionnaires. It asks one question at a time and chooses the most informative
next question from the answers so far. It stops early **only** when a
*conformal risk control* rule certifies the stop. That rule keeps the share of
true positives who are missed at or below a target (5%) in the calibration
population, and the guarantee can be made to hold separately in every
demographic group. The self-harm item (PHQ-9 item 9) is always asked. Any
answer above 0 sends the person through the full questionnaire and shows crisis
contacts.

The repository contains:

| Folder / file | What it is |
|---|---|
| `cas/core.py` | The method: item-masked gradient-boosting models, adaptive question choice, exact curtailment, conformal risk control (marginal, group-wise, PAC) and the baselines |
| `cas/engine.py` | The same rule used live by the web app |
| `study/` | Full, reproducible study on public data: download → harmonise → experiments → tables and figures |
| `app.py`, `templates/` | Flask web app and API |
| `models/cas_artifacts.joblib` | Trained screener produced by `study/run_all.py` |
| `legacy_v1/` | Corrected re-run of the project's first (synthetic-data) experiment, kept for transparency |
| `crisis_resources.json` | Nigerian crisis contacts shown by the app. **Re-check before any deployment** |

## Quick start

```bash
pip install -r requirements.txt
python app.py                  # open http://localhost:5000
```

## Reproduce the study

```bash
python study/download_data.py  # public data, about 40 MB
python study/data_prep.py      # harmonise into one table
python study/run_all.py        # all tables, figures and the app model (about 30 min)
python legacy_v1/train_v1.py   # optional: the original synthetic experiment
```

Results are written to `study/results/` (CSV/JSON tables and `figures/`).
Seeds are fixed, so re-running gives the same numbers.

## Data (all public, open licences)

| Dataset | Use | Licence |
|---|---|---|
| NHANES 2005–2023 PHQ-9 + demographics (US CDC) | Depression development (2005–2016) and temporal validation (2017–2020, 2021–2023) | Public domain |
| Su et al. 2024, *Sci Data* 11:1046: 24,292 Chinese university students, PHQ-9 + GAD-7 ([Zenodo 10423537](https://zenodo.org/records/10423537)) | Anxiety development; external depression test | CC BY 4.0 |
| López Tavera & García: 776 Mexican medical students, PHQ-9 + GAD-7 ([figshare 1305817](https://doi.org/10.6084/m9.figshare.1305817.v3)) | External test, both scales | CC BY 4.0 |
| Hermann 2020: online hobbyist survey, PHQ-9 + GAD-7 ([Mendeley vs8pnynkb3](https://doi.org/10.17632/vs8pnynkb3.1)) | Small external test | CC BY 4.0 |

Raw data are not stored in this repository. They are downloaded by the
script. One source file contains respondent e-mail/IP columns, and the
pipeline never reads them.

## API

`POST /api/next` with `{"module": "phq9" | "gad7", "answers": {"phq2": 1, ...}, "sex": "female", "age_band": "18-39"}`
returns either the next question (`done: false`, `item`, `text`) or the
module result (`done: true`, `decision`, `basis` = exact | certified,
`items_asked`, `safety_flag`).

`POST /api/summary` with `{"phq9": {...}, "gad7": {...}, "sex": ..., "age_band": ...}`
returns the overall risk level, recommendations and crisis contacts.

`GET /api/model-info`, `GET /api/health`.

Answers must be integers 0–3. `sex` and `age_band` are optional (`prefer_not`
falls back to the single, all-respondent threshold).

## Important limitations

* The reference standard is the full PHQ-9/GAD-7 score (≥ 10), not a clinical
  interview. CAS shortens the questionnaire; it does not diagnose.
* The safety guarantee holds for people like those in the calibration data. The
  study shows that it can fail under population shift, and that a modest local
  calibration sample restores it. **Before use in Nigeria, the thresholds must be
  recalibrated on a local sample** (see `study/results/table4_local_recalibration.csv`).
* Not a medical device. Not for clinical use without ethical approval and local validation.

## Licence

Code: MIT (see `LICENSE`). Data remain under their original licences.
