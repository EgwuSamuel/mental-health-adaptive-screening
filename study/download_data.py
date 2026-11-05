"""
Download every public dataset used in the study into study/data/raw/.

Sources (all public, open licence):
  * NHANES 2005-2023 PHQ-9 (DPQ) + demographics (DEMO), US CDC, public domain.
  * Zenodo 10423537 - 24,292 Chinese university students, PHQ-9 + GAD-7 items (CC BY 4.0).
  * Mendeley vs8pnynkb3 - online survey of bonsai hobbyists (uploaded from University of Zululand),
    PHQ-9 + GAD-7 items (CC BY 4.0). The raw file holds emails/IPs: only item columns are read.
  * figshare 1305817 - 776 Mexican medical students, PHQ-9 + GAD-7 items (CC BY 4.0).
  * Zenodo 6454949 - Hospitalised COVID-19 patients, Edo State, Nigeria (CC0).

Run:  python study/download_data.py
"""
from pathlib import Path
import sys
import time
import urllib.request

RAW = Path(__file__).resolve().parent / "data" / "raw"

# NHANES cycle -> (start year, file suffix). P = 2017-March 2020 pre-pandemic
# (it already contains the 2017-2018 "J" participants, so J is not downloaded).
NHANES_CYCLES = {
    "2005-2006": ("2005", "_D"),
    "2007-2008": ("2007", "_E"),
    "2009-2010": ("2009", "_F"),
    "2011-2012": ("2011", "_G"),
    "2013-2014": ("2013", "_H"),
    "2015-2016": ("2015", "_I"),
    "2017-2020": ("2017", "P_"),
    "2021-2023": ("2021", "_L"),
}

OTHER = {
    "china_phq9.csv": "https://zenodo.org/api/records/10423537/files/phq9.csv/content",
    "china_gad7.csv": "https://zenodo.org/api/records/10423537/files/gad7.csv/content",
    "china_demographic.csv": "https://zenodo.org/api/records/10423537/files/demographic.csv/content",
    "hobbyist_phq_gad.xls": (
        "https://data.mendeley.com/public-files/datasets/vs8pnynkb3/files/"
        "022940d6-c6b9-4497-bbaf-74340817228f/file_downloaded"
    ),
    "mexico_medstudents.csv": "https://ndownloader.figshare.com/files/12990629",
    "nigeria_edo.sav": "https://zenodo.org/api/records/6454949/files/Psy_Mani_2022_Data_Spradsheet.sav/content",
    "nigeria_edo_README.doc": "https://zenodo.org/api/records/6454949/files/README__file.doc/content",
}


def nhanes_url(year: str, suffix: str, table: str) -> str:
    name = f"P_{table}" if suffix == "P_" else f"{table}{suffix}"
    return f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt", name


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-download-script"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if data[:15].lower().startswith(b"<!doctype html") or data[:5].lower() == b"<html":
                raise RuntimeError("got an HTML page instead of a data file")
            dest.write_bytes(data)
            print(f"  got  {dest.name} ({len(data)/1e6:.2f} MB)")
            return
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt + 1} for {dest.name}: {e}")
            time.sleep(3)
    sys.exit(f"Failed to download {url}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print("NHANES")
    for cycle, (year, suffix) in NHANES_CYCLES.items():
        for table in ("DPQ", "DEMO"):
            url, name = nhanes_url(year, suffix, table)
            fetch(url, RAW / f"nhanes_{name}.xpt")
    print("Other public datasets")
    for name, url in OTHER.items():
        fetch(url, RAW / name)
    print("Done ->", RAW)


if __name__ == "__main__":
    main()
