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

        if game == "Melate" and len(nums) >= 7:
            nums = nums[:-1]

        nums = nums[:6]
        nums_int = sorted(int(n) for n in nums if n.isdigit())
        nums_fmt = [str(n).zfill(2) for n in nums_int]

        while len(nums_fmt) < 6:
            nums_fmt.append("")

        row = {
            "FECHA": r["_fecha_dt"].strftime("%d/%m/%Y"),
            "N1": nums_fmt[0],
            "N2": nums_fmt[1],
            "N3": nums_fmt[2],
            "N4": nums_fmt[3],
            "N5": nums_fmt[4],
            "N6": nums_fmt[5],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def update_local_csv(game, df_norm):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{game.lower()}.csv")

    if not os.path.exists(file_path):
        logger.warning("%s no existe. Creando nuevo archivo.", file_path)
        df_norm.to_csv(file_path, index=False)

        for _, row in df_norm.iterrows():
            fecha_iso = pd.to_datetime(row["FECHA"], dayfirst=True).strftime("%Y-%m-%d")
            nums = [int(row[f"N{i}"]) for i in range(1,7) if row[f"N{i}"].isdigit()]
            if len(nums) == 6:
                insert_draw(game, fecha_iso, nums)

        return len(df_norm)

    existing_df = pd.read_csv(file_path)
    existing_df["_fecha_dt"] = pd.to_datetime(existing_df["FECHA"], dayfirst=True, errors="coerce")
    last_date = existing_df["_fecha_dt"].max()

    df_norm["_fecha_dt"] = pd.to_datetime(df_norm["FECHA"], dayfirst=True, errors="coerce")

    if last_date is not None and not pd.isna(last_date):
        new_df = df_norm[df_norm["_fecha_dt"] > last_date]
    else:
        new_df = df_norm

    if new_df.empty:
        logger.info("%s: no hay filas nuevas.", game)
        return 0

    for _, row in new_df.iterrows():
        fecha_iso = pd.to_datetime(row["FECHA"], dayfirst=True).strftime("%Y-%m-%d")
        nums = [int(row[f"N{i}"]) for i in range(1,7) if row[f"N{i}"].isdigit()]
        if len(nums) == 6:
            insert_draw(game, fecha_iso, nums)

    updated_df = pd.concat([df_existing, df_new])

# ORDENAR CRONOLÓGICAMENTE (MAS RECIENTE ARRIBA)
updated_df = updated_df.sort_values("_fecha_dt", ascending=False)

# eliminar columna auxiliar
updated_df = updated_df.drop(columns=["_fecha_dt"], errors="ignore")

# guardar csv
updated_df.to_csv(csv_path, index=False)

    logger.info("%s: agregadas %d filas.", game, len(new_df))
    return len(new_df)


def send_email_summary(text_body):
    user = os.getenv("EMAIL_USER")
    pw = os.getenv("EMAIL_PASS")
    to = os.getenv("EMAIL_TO") or user

    if not user or not pw or not _HAS_YAGMAIL:
        return

    yag = yagmail.SMTP(user, pw)
    yag.send(to, "MDV - Actualización resultados", text_body)
    logger.info("Correo enviado a %s", to)


def main():
    logger.info("Iniciando actualización local de los 3 juegos...")

    initialize_database()

    summary = []

    for game, url in URLS.items():
        try:
            text = download_csv_text(url)
            df = parse_csv_generic(text, game)
            inserted = update_local_csv(game, df)
            summary.append(f"{game}: {inserted} filas nuevas")
        except Exception as e:
            logger.exception("Error en %s", game)
            summary.append(f"{game}: ERROR {e}")

    report = "\n".join(summary)
    logger.info("Resumen:\n%s", report)

    send_email_summary(report)

    logger.info("Proceso finalizado.")


if __name__ == "__main__":
    main()

df = df.sort_values("FECHA", ascending=False)
df.to_csv(path, index=False)
