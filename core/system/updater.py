#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_updater.py
Descarga los resultados del sitio oficial de Lotería Nacional,
compara con el CSV local y agrega solo las filas nuevas.
Sin dependencia de base de datos.
"""

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

# ── Rutas ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── URLs oficiales ───────────────────────────────────────────────────
URLS = {
    "Melate":     "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=TQBlAGwAYQB0AGUA",
    "Revancha":   "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=UgBlAHYAYQBuAGMAaABhAA==",
    "Revanchita": "https://www.loterianacional.gob.mx/Home/Historicos?ARHP=UgBlAHYAYQBuAGMAaABpAHQAYQA=",
}

# ── Columnas esperadas por juego ─────────────────────────────────────
HEADERS = {
    "Melate":     ["FECHA", "N1", "N2", "N3", "N4", "N5", "N6", "BONO"],
    "Revancha":   ["FECHA", "N1", "N2", "N3", "N4", "N5", "N6"],
    "Revanchita": ["FECHA", "N1", "N2", "N3", "N4", "N5", "N6"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("csv_updater")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ── Descarga ─────────────────────────────────────────────────────────

def download_csv_text(url: str, timeout: int = 30) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
    resp.raise_for_status()
    return resp.text


# ── Parseo ───────────────────────────────────────────────────────────

def parse_raw(text: str, game: str) -> pd.DataFrame:
    """Convierte el texto descargado del sitio oficial en un DataFrame limpio."""
    try:
        df = pd.read_csv(StringIO(text), encoding="latin1", engine="python")
    except Exception:
        df = pd.read_csv(StringIO(text), encoding="utf-8",
                         engine="python", on_bad_lines="skip")

    df = df.dropna(axis=1, how="all")

    # Detectar columna de fecha
    date_col = next(
        (c for c in df.columns if "fecha" in str(c).lower()),
        df.columns[0]
    )
    df["_fecha_dt"] = pd.to_datetime(
        df[date_col].astype(str), dayfirst=True, errors="coerce"
    )

    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["_fecha_dt"]):
            continue

        # Extraer todos los números de 1-2 dígitos de la fila
        vals = [str(x).strip() for x in r if pd.notna(x)]
        nums = [n for n in vals if re.fullmatch(r"\d{1,2}", n)]

        bono = ""
        if game == "Melate" and len(nums) >= 7:
            bono = nums[-1]          # último número = bono
            nums = nums[:-1]

        nums = nums[:6]
        nums_int = sorted(int(n) for n in nums if n.isdigit())

        # Validar que todos los números estén en rango 1-56
        if len(nums_int) != 6 or any(n < 1 or n > 56 for n in nums_int):
            continue

        row = {"FECHA": r["_fecha_dt"].strftime("%d/%m/%Y")}
        for i, n in enumerate(nums_int, 1):
            row[f"N{i}"] = n
        if game == "Melate":
            row["BONO"] = int(bono) if bono and bono.isdigit() else ""

        rows.append(row)

    return pd.DataFrame(rows)


# ── Actualización del CSV local ───────────────────────────────────────

def update_csv(game: str, df_new: pd.DataFrame) -> int:
    """
    Compara df_new con el CSV local.
    Agrega solo las filas más recientes que la última fecha registrada.
    Guarda siempre con la fila más reciente primero.
    Retorna el número de filas agregadas.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{game.lower()}.csv")

    df_new["_fecha_dt"] = pd.to_datetime(
        df_new["FECHA"], dayfirst=True, errors="coerce"
    )

    # Si el CSV no existe, crear desde cero
    if not os.path.exists(csv_path):
        logger.warning("[%s] CSV no existe, creando desde cero.", game)
        df_out = df_new.sort_values("_fecha_dt", ascending=False)
        df_out = df_out.drop(columns=["_fecha_dt"])
        df_out.to_csv(csv_path, index=False)
        logger.info("[%s] Creado con %d filas.", game, len(df_out))
        return len(df_out)

    # Leer CSV existente
    df_existing = pd.read_csv(csv_path)
    df_existing["_fecha_dt"] = pd.to_datetime(
        df_existing["FECHA"], dayfirst=True, errors="coerce"
    )

    # Fecha más reciente que ya tenemos
    last_date = df_existing["_fecha_dt"].max()

    if pd.isna(last_date):
        logger.warning("[%s] No se pudo leer la fecha del CSV. Reconstruyendo.", game)
        df_out = df_new.sort_values("_fecha_dt", ascending=False)
        df_out = df_out.drop(columns=["_fecha_dt"])
        df_out.to_csv(csv_path, index=False)
        return len(df_out)

    # Filtrar solo filas más nuevas
    df_add = df_new[df_new["_fecha_dt"] > last_date].copy()

    if df_add.empty:
        logger.info("[%s] Sin filas nuevas. Último registro: %s",
                    game, last_date.strftime("%d/%m/%Y"))
        return 0

    # Combinar, ordenar y guardar
    df_combined = pd.concat([df_existing, df_add], ignore_index=True)
    df_combined = df_combined.sort_values("_fecha_dt", ascending=False)
    df_combined = df_combined.drop_duplicates(subset=["FECHA"])
    df_combined = df_combined.drop(columns=["_fecha_dt"])

    df_combined.to_csv(csv_path, index=False)
    logger.info("[%s] Agregadas %d filas nuevas. Total: %d",
                game, len(df_add), len(df_combined))
    return len(df_add)


# ── Email resumen ─────────────────────────────────────────────────────

def send_email_summary(body: str):
    user = os.getenv("EMAIL_USER")
    pw   = os.getenv("EMAIL_PASS")
    to   = os.getenv("EMAIL_TO") or user
    if not user or not pw or not _HAS_YAGMAIL:
        logger.warning("Email no configurado, omitiendo resumen.")
        return
    try:
        yag = yagmail.SMTP(user, pw)
        yag.send(to, "MDV — Actualización históricos", body)
        logger.info("Resumen enviado a %s", to)
    except Exception as e:
        logger.error("Error enviando email: %s", e)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    logger.info("═" * 50)
    logger.info("Iniciando actualización de históricos...")
    summary = []

    for game, url in URLS.items():
        try:
            logger.info("[%s] Descargando del sitio oficial...", game)
            text = download_csv_text(url)

            logger.info("[%s] Parseando...", game)
            df = parse_raw(text, game)

            if df.empty:
                logger.warning("[%s] Sin datos en la descarga.", game)
                summary.append(f"{game}: sin datos descargados")
                continue

            inserted = update_csv(game, df)
            summary.append(f"{game}: {inserted} filas nuevas")

        except Exception as e:
            logger.exception("[%s] Error: %s", game, e)
            summary.append(f"{game}: ERROR — {e}")

    report = "\n".join(summary)
    logger.info("Resumen:\n%s", report)
    logger.info("═" * 50)
    send_email_summary(report)
    logger.info("Proceso finalizado.")


if __name__ == "__main__":
    main()
