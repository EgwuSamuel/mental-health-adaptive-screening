"""
Harmonise every public dataset into one tidy format:

    dataset, cycle, sex, age, age_band, race, phq1..phq9, gad1..gad7

Items are 0-3. Missing / refused / don't-know answers make a record incomplete
for that instrument. Only the item columns of the hobbyist survey are read
(the raw file also holds e-mail and IP addresses, which are never loaded).

Run:  python study/data_prep.py
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "processed"

PHQ = [f"phq{i}" for i in range(1, 10)]
GAD = [f"gad{i}" for i in range(1, 8)]
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 5: "Other/multiracial"}
NHANES_FILES = {
    "2005-2006": "_D", "2007-2008": "_E", "2009-2010": "_F", "2011-2012": "_G",
    "2013-2014": "_H", "2015-2016": "_I", "2017-2020": "P_", "2021-2023": "_L",
}


def age_band(a):
    return pd.cut(a, [0, 39, 59, 200], labels=["18-39", "40-59", "60+"]).astype(str)


def clean_items(df, cols):
    x = df[cols].astype(float).round()
    x = x.where(x.isin([0, 1, 2, 3]))  # 7/9 (refused / don't know) -> missing
    return x


def load_nhanes():
    frames = []
    for cycle, suf in NHANES_FILES.items():
        dpq_name = f"P_DPQ" if suf == "P_" else f"DPQ{suf}"
        demo_name = f"P_DEMO" if suf == "P_" else f"DEMO{suf}"
        dpq = pd.read_sas(RAW / f"nhanes_{dpq_name}.xpt")
        demo = pd.read_sas(RAW / f"nhanes_{demo_name}.xpt")
        d = dpq.merge(demo[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1"]], on="SEQN")
        d = d.rename(columns={f"DPQ0{i}0": f"phq{i}" for i in range(1, 10)})
        d[PHQ] = clean_items(d, PHQ)
        d = d[d.RIDAGEYR >= 18]
        frames.append(pd.DataFrame({
            "dataset": "NHANES", "cycle": cycle, "id": d.SEQN.astype(int).astype(str),
            "sex": d.RIAGENDR.map({1: "male", 2: "female"}),
            "age": d.RIDAGEYR, "race": d.RIDRETH1.map(RACE), **{c: d[c] for c in PHQ}}))
    return pd.concat(frames, ignore_index=True)


def load_china():
    p = pd.read_csv(RAW / "china_phq9.csv")
    g = pd.read_csv(RAW / "china_gad7.csv")
    dem = pd.read_csv(RAW / "china_demographic.csv")
    p = p.rename(columns={f"question{i}": f"phq{i}" for i in range(1, 10)})
    p = p.rename(columns={f"time{i}": f"phq_time{i}" for i in range(1, 10)})
    g = g.rename(columns={f"question{i}": f"gad{i}" for i in range(1, 8)})
    g = g.rename(columns={f"time{i}": f"gad_time{i}" for i in range(1, 8)})
    d = dem.merge(p.drop(columns="score"), on="export_id").merge(g.drop(columns="score"), on="export_id")
    d[PHQ] = clean_items(d, PHQ)
    d[GAD] = clean_items(d, GAD)
    times = [c for c in d.columns if "_time" in c]
    return pd.DataFrame({"dataset": "China students", "cycle": "2021", "id": d.export_id.astype(str),
                         "sex": d.gender.str.lower(), "age": d.age, "race": "Chinese",
                         **{c: d[c] for c in PHQ + GAD + times}})


def load_mexico():
    d = pd.read_csv(RAW / "mexico_medstudents.csv")
    d[PHQ] = clean_items(d, PHQ)
    d[GAD] = clean_items(d, GAD)
    sex = d.gender.astype(str).str.lower().map(
        {"f": "female", "female": "female", "m": "male", "male": "male", "1": "male", "2": "female",
         "0": "female"})
    return pd.DataFrame({"dataset": "Mexico medical students", "cycle": "2014", "id": d.id.astype(str),
                         "sex": sex, "age": d.age, "race": "Mexican", **{c: d[c] for c in PHQ + GAD}})


def load_hobbyist():
    # Read ONLY the age/gender and item columns; skip identifying columns (e-mail, IP).
    raw = pd.read_excel(RAW / "hobbyist_phq_gad.xls", sheet_name="RAW", header=None,
                        usecols=[14, 15] + list(range(139, 146)) + list(range(147, 156)))
    body = raw.iloc[3:].reset_index(drop=True)
    body.columns = ["age_raw", "sex_raw"] + GAD + PHQ
    for c in PHQ + GAD:
        body[c] = pd.to_numeric(body[c], errors="coerce")
    # detect 1-4 coding and shift to 0-3 if needed
    if body[PHQ + GAD].min().min() >= 1 and body[PHQ + GAD].max().max() == 4:
        body[PHQ + GAD] = body[PHQ + GAD] - 1
    body[PHQ] = clean_items(body, PHQ)
    body[GAD] = clean_items(body, GAD)
    return pd.DataFrame({"dataset": "Online hobbyist survey", "cycle": "2013",
                         "id": body.index.astype(str),
                         "sex": pd.to_numeric(body.sex_raw, errors="coerce").map({1: "male", 2: "female"}),
                         "age": np.nan, "race": "unknown", **{c: body[c] for c in PHQ + GAD}}), body


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hob, hob_body = load_hobbyist()
    print("hobbyist age/sex raw values:", hob_body.age_raw.value_counts().head(8).to_dict(),
          hob_body.sex_raw.value_counts().head(5).to_dict())
    data = pd.concat([load_nhanes(), load_china(), load_mexico(), hob], ignore_index=True)
    data["age_band"] = np.where(data.age.notna(), age_band(data.age.fillna(0)), "unknown")
    data["sex"] = data.sex.fillna("unknown")
    data["phq_complete"] = data[PHQ].notna().all(axis=1)
    data["gad_complete"] = data[GAD].notna().all(axis=1) if set(GAD) <= set(data.columns) else False
    data["phq_total"] = data[PHQ].sum(axis=1, min_count=9)
    data["gad_total"] = data[GAD].sum(axis=1, min_count=7)
    data.to_pickle(OUT / "all_data.pkl")

    summ = (data.groupby(["dataset", "cycle"])
            .agg(n=("id", "size"), phq_complete=("phq_complete", "sum"), gad_complete=("gad_complete", "sum"),
                 dep_pos=("phq_total", lambda s: float((s >= 10).sum())),
                 anx_pos=("gad_total", lambda s: float((s >= 10).sum())))
            .reset_index())
    summ["dep_prev_%"] = (100 * summ.dep_pos / summ.phq_complete.replace(0, np.nan)).round(1)
    summ["anx_prev_%"] = (100 * summ.anx_pos / summ.gad_complete.replace(0, np.nan)).round(1)
    print(summ.to_string(index=False))
    print(data.groupby("dataset").age_band.value_counts().unstack(fill_value=0))
    print(data.groupby("dataset").sex.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
