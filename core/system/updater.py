import requests
import pandas as pd
from io import StringIO
from system.database import initialize_database, insert_draw

URLS = {
    "Melate": "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUA",
    "Revancha": "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=UgBlAHYAYQBuAGMAaABhAA==",
    "Revanchita": "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=UgBlAHYAYQBuAGMAaABpAHQAYQA="
}

def parse_csv(text):
    df = pd.read_csv(StringIO(text), encoding="latin1", engine="python")
    df = df.dropna(axis=1, how="all")
    df["_fecha_dt"] = pd.to_datetime(df.iloc[:,0], dayfirst=True, errors="coerce")
    return df

def update_all():
    initialize_database()
    for game,url in URLS.items():
        r = requests.get(url, timeout=30, verify=False)
        r.raise_for_status()
        df = parse_csv(r.text)

        for _,row in df.iterrows():
            if pd.isna(row["_fecha_dt"]):
                continue

            fecha = row["_fecha_dt"].strftime("%Y-%m-%d")
            nums = [int(x) for x in row if str(x).isdigit()][:6]
            nums = sorted(nums)

            if len(nums)==6:
                insert_draw(game,fecha,nums)
