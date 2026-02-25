# csv_updater.py

from system.database import initialize_database, insert_draw

import os
import re
import logging
from io import StringIO
import urllib3

import requests
import pandas as pd

try:
    import yagmail
    _HAS_YAGMAIL = True
except Exception:
    _HAS_YAGMAIL = False


URLS = {
    "Melate": "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUA",
    "Revancha": "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=UgBlAHYAYQBuAGMAaABhAA==",
    "Revanchita": "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=UgBlAHYAYQBuAGMAaABpAHQAYQA="
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("csv_updater")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_csv_text(url, timeout=20):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
    resp.raise_for_status()
    return resp.text


def parse_csv_generic(text, game):
    try:
        df = pd.read_csv(StringIO(text), encoding="latin1", engine="python")
    except Exception:
        df = pd.read_csv(StringIO(text), encoding="utf-8", engine="python", on_bad_lines="skip")

    df = df.dropna(axis=1, how="all")

    date_col = None
    for c in df.columns:
        if "fecha" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df["_fecha_dt"] = pd.to_datetime(df[date_col].astype(str), dayfirst=True, errors="coerce")

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["_fecha_dt"]):
            continue

        vals = [str(x).strip() for x in r if pd.notna(x)]
        nums = [n for n in vals if re.fullmatch(r"\d{1,2}", n)]

        bono = ""
        if game == "Melate" and len(nums) >= 7:
            bono = nums[-1]
            nums = nums[:-1]

        nums = nums[:6]
        nums_int = sorted(int(n) for n in nums if n.isdigit())
        nums_fmt = [str(n).zfill(2) for n in nums_int]

        while len(nums_fmt) < 6:
            nums_fmt.append("")

        row = {"FECHA": r["_fecha_dt"]

