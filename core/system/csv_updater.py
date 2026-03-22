#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_updater.py
Descarga los historicos oficiales de Loteria Nacional y actualiza
los CSV locales con solo las filas nuevas.

FUENTE OFICIAL:
  https://www.loterianacional.gob.mx/Documentos/Historicos/Melate.csv
  https://www.loterianacional.gob.mx/Documentos/Historicos/Revancha.csv
  https://www.loterianacional.gob.mx/Documentos/Historicos/Revanchita.csv

ESTRUCTURA CSV OFICIAL (Melate):
  NPRODUCTO, CONCURSO, R1, R2, R3, R4, R5, R6, R7, BOLSA, FECHA
  - R1-R6: numeros principales en orden ascendente
  - R7:    BONO (solo Melate)
  - BOLSA: pozo acumulado en pesos
  - CONCURSO: numero de sorteo

ESTRUCTURA CSV LOCAL:
  FECHA, CONCURSO, N1, N2, N3, N4, N5, N6, BONO, BOLSA  (Melate)
  FECHA, CONCURSO, N1, N2, N3, N4, N5, N6, BOLSA         (Revancha/Revanchita)
"""

import os
import logging
import urllib3
import requests
import pandas as pd
from io import StringIO

try:
    import yagmail
    _HAS_YAGMAIL = True
except Exception:
    _HAS_YAGMAIL = False

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(os.path.dirname(_HERE))
_DATA_DIR = os.path.join(_ROOT, "data")

URLS = {
    "Melate":     "https://www.loterianacional.gob.mx/Documentos/Historicos/Melate.csv",
    "Revancha":   "https://www.loterianacional.gob.mx/Documentos/Historicos/Revancha.csv",
    "Revanchita": "https://www.loterianacional.gob.mx/Documentos/Historicos/Revanchita.csv",
}

GAME_CONFIG = {
    "Melate": {
        "has_bono": True,
        "n_balls":  6,
        "bono_col": "R7",
        "local_cols": ["FECHA","CONCURSO","N1","N2","N3","N4","N5","N6","BONO","BOLSA"],
        },
    "Revanchita": {
        "has_bono": False,
        "n_balls":  6,
        "bono_col": None,
        "ball_prefix": "F",
        "local_cols": ["FECHA","CONCURSO","N1","N2","N3","N4","N5","N6","BOLSA"],
    },
        "Revanchita": {
        "has_bono": False,
        "n_balls":  6,
        "bono_col": None,
        "ball_prefix": "F",
        "local_cols": ["FECHA","CONCURSO","N1","N2","N3","N4","N5","N6","BOLSA"],
    },

}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("csv_updater")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_official_csv(url: str, timeout: int = 60) -> pd.DataFrame:
    """Descarga el CSV oficial y retorna un DataFrame con los datos crudos."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    logger.info(f"Descargando: {url}")
    resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
    resp.raise_for_status()

    # Intentar encodings comunes del sitio
    for enc in ["latin1", "utf-8", "cp1252"]:
        try:
            text = resp.content.decode(enc)
            df = pd.read_csv(StringIO(text), sep=None, engine="python")
            if not df.empty:
                logger.info(f"Descargado correctamente ({len(df)} filas, enc={enc})")
                return df
        except Exception:
            continue

    raise RuntimeError("No se pudo parsear el CSV oficial")


def parse_official(df_raw: pd.DataFrame, game: str) -> pd.DataFrame:
    """
    Convierte el DataFrame crudo del sitio oficial al formato local.
    Columnas oficiales: NPRODUCTO, CONCURSO, R1-R6, [R7], BOLSA, FECHA
    Columnas locales:   FECHA, CONCURSO, N1-N6, [BONO], BOLSA
    """
    cfg = GAME_CONFIG[game]

    # Normalizar nombres de columnas (quitar espacios, mayusculas)
    df_raw.columns = [str(c).strip().upper() for c in df_raw.columns]
    logger.info(f"[{game}] Columnas en CSV oficial: {list(df_raw.columns)}")

    # Validar columnas requeridas
    required = ["CONCURSO", "R1", "R2", "R3", "R4", "R5", "R6", "BOLSA", "FECHA"]
    for col in required:
        if col not in df_raw.columns:
            raise RuntimeError(f"Columna faltante en CSV oficial de {game}: {col}")

    rows = []
    n_invalid = 0

    for _, row in df_raw.iterrows():
        try:
            # Fecha
            fecha_dt = pd.to_datetime(str(row["FECHA"]), dayfirst=True, errors="coerce")
            if pd.isna(fecha_dt):
                n_invalid += 1
                continue

            # Numero de sorteo
            concurso = int(row["CONCURSO"])

            #            # Numeros principales R1-R6
            nums = []
            prefix = cfg.get("ball_prefix", "R")
            for i in range(1, cfg["n_balls"] + 1):
                col = f"{prefix}{i}"
                v = int(row[col])
                if not (1 <= v <= 56):
                    raise ValueError(f"Numero fuera de rango: {v}")
                nums.append(v)

            # BONO (R7, solo Melate)
            bono = None
            if cfg["has_bono"] and cfg["bono_col"] and cfg["bono_col"] in df_raw.columns:
                bv = row[cfg["bono_col"]]
                if pd.notna(bv):
                    bono = int(bv)

            # BOLSA
            bolsa = None
            if pd.notna(row.get("BOLSA")):
                try:
                    bolsa = int(float(str(row["BOLSA"]).replace(",", "")))
                except Exception:
                    bolsa = None

            # Construir fila local
            local_row = {
                "FECHA":    fecha_dt.strftime("%d/%m/%Y"),
                "CONCURSO": concurso,
                "N1": nums[0], "N2": nums[1], "N3": nums[2],
                "N4": nums[3], "N5": nums[4], "N6": nums[5],
            }
            if cfg["has_bono"]:
                local_row["BONO"] = bono if bono is not None else ""
            local_row["BOLSA"] = bolsa if bolsa is not None else ""

            rows.append(local_row)

        except Exception as e:
            n_invalid += 1
            continue

    if n_invalid > 0:
        logger.warning(f"[{game}] {n_invalid} filas invalidas ignoradas")

    df_out = pd.DataFrame(rows)
    df_out["_fecha_dt"] = pd.to_datetime(df_out["FECHA"], dayfirst=True, errors="coerce")
    df_out = df_out.sort_values("_fecha_dt", ascending=False).drop(columns=["_fecha_dt"])
    df_out = df_out.reset_index(drop=True)

    logger.info(f"[{game}] {len(df_out)} filas validas parseadas")
    return df_out


def update_local_csv(game: str, df_new: pd.DataFrame) -> int:
    """
    Compara df_new con el CSV local y agrega solo las filas nuevas.
    Maneja migracion del formato antiguo (sin CONCURSO/BOLSA) al nuevo.
    Retorna el numero de filas agregadas.
    """
    os.makedirs(_DATA_DIR, exist_ok=True)
    csv_path = os.path.join(_DATA_DIR, f"{game.lower()}.csv")
    cfg = GAME_CONFIG[game]

    df_new["_fecha_dt"] = pd.to_datetime(df_new["FECHA"], dayfirst=True, errors="coerce")

    # CSV no existe: crear desde cero
    if not os.path.exists(csv_path):
        logger.warning(f"[{game}] CSV no existe, creando desde cero con {len(df_new)} filas")
        df_out = df_new.drop(columns=["_fecha_dt"])
        df_out.to_csv(csv_path, index=False)
        return len(df_out)

    # CSV existe: leer y comparar
    df_existing = pd.read_csv(csv_path)

    # Detectar si el CSV existente tiene el formato antiguo (sin CONCURSO/BOLSA)
    is_old_format = "CONCURSO" not in df_existing.columns

    if is_old_format:
        logger.info(f"[{game}] Detectado formato antiguo. Migrando al nuevo formato...")
        # Reemplazar completamente con los datos oficiales nuevos
        df_out = df_new.drop(columns=["_fecha_dt"])
        df_out.to_csv(csv_path, index=False)
        logger.info(f"[{game}] Migrado: {len(df_out)} filas guardadas con nuevo formato")
        return len(df_out)

    # Formato nuevo: agregar solo filas nuevas por numero de CONCURSO
    existing_concursos = set(df_existing["CONCURSO"].dropna().astype(int).tolist())
    df_add = df_new[~df_new.apply(
        lambda r: int(r["CONCURSO"]) in existing_concursos, axis=1
    )].copy()

    if df_add.empty:
        # Verificar por fecha como respaldo
        df_existing["_fecha_dt"] = pd.to_datetime(
            df_existing["FECHA"], dayfirst=True, errors="coerce"
        )
        last_date = df_existing["_fecha_dt"].max()
        df_existing = df_existing.drop(columns=["_fecha_dt"])
        if not pd.isna(last_date):
            df_add = df_new[df_new["_fecha_dt"] > last_date].copy()

    if df_add.empty:
        logger.info(f"[{game}] Sin filas nuevas. CSV actualizado.")
        return 0

    # Combinar, ordenar y guardar
    df_add_clean = df_add.drop(columns=["_fecha_dt"])
    df_combined  = pd.concat([df_existing, df_add_clean], ignore_index=True)
    df_combined["_fecha_dt"] = pd.to_datetime(
        df_combined["FECHA"], dayfirst=True, errors="coerce"
    )
    df_combined = (df_combined
                   .sort_values("_fecha_dt", ascending=False)
                   .drop_duplicates(subset=["CONCURSO"])
                   .drop(columns=["_fecha_dt"])
                   .reset_index(drop=True))
    df_combined.to_csv(csv_path, index=False)

    logger.info(f"[{game}] Agregadas {len(df_add_clean)} filas nuevas. Total: {len(df_combined)}")
    return len(df_add_clean)


def send_email_summary(body: str):
    user = os.getenv("EMAIL_USER")
    pw   = os.getenv("EMAIL_PASS")
    to   = os.getenv("EMAIL_TO") or user
    if not user or not pw or not _HAS_YAGMAIL:
        logger.warning("Email no configurado, omitiendo resumen.")
        return
    try:
        yag = yagmail.SMTP(user, pw)
        yag.send(to, "MDV - Actualizacion historicos", body)
        logger.info(f"Resumen enviado a {to}")
    except Exception as e:
        logger.error(f"Error enviando email: {e}")


def main():
    logger.info("=" * 55)
    logger.info("Iniciando actualizacion de historicos oficiales...")
    summary = []

    for game, url in URLS.items():
        try:
            df_raw    = download_official_csv(url)
            df_parsed = parse_official(df_raw, game)
            inserted  = update_local_csv(game, df_parsed)
            summary.append(f"{game}: {inserted} filas nuevas")
        except Exception as e:
            logger.exception(f"[{game}] Error: {e}")
            summary.append(f"{game}: ERROR - {e}")

    report = "\n".join(summary)
    logger.info(f"Resumen:\n{report}")
    logger.info("=" * 55)
    send_email_summary(report)
    logger.info("Proceso finalizado.")


if __name__ == "__main__":
    main()