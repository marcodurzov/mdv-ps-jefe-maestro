#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_updater.py
Descarga los historicos oficiales de Loteria Nacional y actualiza
los CSV locales con solo las filas nuevas.

ESTRUCTURA CSV OFICIAL:
  Melate:     NPRODUCTO, CONCURSO, R1, R2, R3, R4, R5, R6, R7, BOLSA, FECHA
  Revancha:   PRODUCTO,  CONCURSO, R1, R2, R3, R4, R5, R6,     BOLSA, FECHA
  Revanchita: NPRODUCTO, CONCURSO, F1, F2, F3, F4, F5, F6,     BOLSA, FECHA

  Melate R1-R6 = numeros principales, R7 = BONO
  Revancha R1-R6 = numeros principales, sin BONO
  Revanchita F1-F6 = numeros principales, sin BONO

ESTRUCTURA CSV LOCAL GENERADO:
  Melate:                FECHA, CONCURSO, N1-N6, BONO, BOLSA
  Revancha/Revanchita:   FECHA, CONCURSO, N1-N6, BOLSA
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

# Configuracion por juego basada en la estructura real de cada CSV oficial
GAME_CONFIG = {
    "Melate": {
        "has_bono":    True,
        "ball_cols":   ["R1", "R2", "R3", "R4", "R5", "R6"],
        "bono_col":    "R7",
        "local_cols":  ["FECHA", "CONCURSO", "N1", "N2", "N3", "N4", "N5", "N6", "BONO", "BOLSA"],
    },
    "Revancha": {
        "has_bono":    False,
        "ball_cols":   ["R1", "R2", "R3", "R4", "R5", "R6"],
        "bono_col":    None,
        "local_cols":  ["FECHA", "CONCURSO", "N1", "N2", "N3", "N4", "N5", "N6", "BOLSA"],
    },
    "Revanchita": {
        "has_bono":    False,
        "ball_cols":   ["F1", "F2", "F3", "F4", "F5", "F6"],
        "bono_col":    None,
        "local_cols":  ["FECHA", "CONCURSO", "N1", "N2", "N3", "N4", "N5", "N6", "BOLSA"],
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("csv_updater")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_official_csv(url, timeout=60):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    logger.info("Descargando: %s", url)
    resp = requests.get(url, headers=headers, timeout=timeout, verify=False)
    resp.raise_for_status()
    for enc in ["latin1", "utf-8", "cp1252"]:
        try:
            text = resp.content.decode(enc)
            df = pd.read_csv(StringIO(text), sep=None, engine="python")
            if not df.empty:
                logger.info("Descargado OK (%d filas, enc=%s)", len(df), enc)
                return df
        except Exception:
            continue
    raise RuntimeError("No se pudo parsear el CSV oficial")


def parse_official(df_raw, game):
    cfg = GAME_CONFIG[game]
    df_raw.columns = [str(c).strip().upper() for c in df_raw.columns]
    logger.info("[%s] Columnas oficiales: %s", game, list(df_raw.columns))

    # Validar columnas requeridas
    required = ["CONCURSO", "BOLSA", "FECHA"] + cfg["ball_cols"]
    for col in required:
        if col not in df_raw.columns:
            raise RuntimeError("Columna faltante en CSV oficial de %s: %s" % (game, col))

    rows = []
    n_invalid = 0

    for _, row in df_raw.iterrows():
        try:
            fecha_dt = pd.to_datetime(str(row["FECHA"]), dayfirst=True, errors="coerce")
            if pd.isna(fecha_dt):
                n_invalid += 1
                continue

            concurso = int(row["CONCURSO"])

            nums = []
            for col in cfg["ball_cols"]:
                v = int(row[col])
                if not (1 <= v <= 56):
                    raise ValueError("Numero fuera de rango: %d" % v)
                nums.append(v)

            bono = None
            if cfg["has_bono"] and cfg["bono_col"]:
                bv = row.get(cfg["bono_col"])
                if pd.notna(bv):
                    bono = int(bv)

            bolsa = None
            raw_bolsa = row.get("BOLSA")
            if pd.notna(raw_bolsa):
                try:
                    bolsa = int(float(str(raw_bolsa).replace(",", "")))
                except Exception:
                    bolsa = None

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

        except Exception:
            n_invalid += 1
            continue

    if n_invalid > 0:
        logger.warning("[%s] %d filas invalidas ignoradas", game, n_invalid)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        raise RuntimeError("Sin filas validas para %s" % game)

    df_out["_fecha_dt"] = pd.to_datetime(df_out["FECHA"], dayfirst=True, errors="coerce")
    df_out = (df_out
              .sort_values("_fecha_dt", ascending=False)
              .drop(columns=["_fecha_dt"])
              .reset_index(drop=True))

    logger.info("[%s] %d filas validas parseadas", game, len(df_out))
    return df_out


def update_local_csv(game, df_new):
    os.makedirs(_DATA_DIR, exist_ok=True)
    csv_path = os.path.join(_DATA_DIR, "%s.csv" % game.lower())

    df_new = df_new.copy()
    df_new["_fecha_dt"] = pd.to_datetime(df_new["FECHA"], dayfirst=True, errors="coerce")

    # CSV no existe: crear desde cero
    if not os.path.exists(csv_path):
        logger.warning("[%s] CSV no existe, creando desde cero", game)
        df_out = df_new.drop(columns=["_fecha_dt"])
        df_out.to_csv(csv_path, index=False)
        logger.info("[%s] Creado con %d filas", game, len(df_out))
        return len(df_out)

    df_existing = pd.read_csv(csv_path)

    # Migrar formato antiguo si no tiene CONCURSO
    if "CONCURSO" not in df_existing.columns:
        logger.info("[%s] Formato antiguo detectado, migrando...", game)
        df_out = df_new.drop(columns=["_fecha_dt"])
        df_out.to_csv(csv_path, index=False)
        logger.info("[%s] Migrado: %d filas", game, len(df_out))
        return len(df_out)

    # Formato nuevo: agregar solo filas con CONCURSO nuevo
    existing_concursos = set(
        df_existing["CONCURSO"].dropna().astype(int).tolist()
    )
    df_add = df_new[
        df_new["CONCURSO"].apply(lambda x: int(x) not in existing_concursos)
    ].copy()

    if df_add.empty:
        logger.info("[%s] Sin filas nuevas", game)
        return 0

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

    logger.info("[%s] Agregadas %d filas. Total: %d", game, len(df_add_clean), len(df_combined))
    return len(df_add_clean)


def send_email_summary(body):
    user = os.getenv("EMAIL_USER")
    pw   = os.getenv("EMAIL_PASS")
    to   = os.getenv("EMAIL_TO") or user
    if not user or not pw or not _HAS_YAGMAIL:
        logger.warning("Email no configurado")
        return
    try:
        yag = yagmail.SMTP(user, pw)
        yag.send(to, "MDV - Actualizacion historicos", body)
        logger.info("Resumen enviado a %s", to)
    except Exception as e:
        logger.error("Error enviando email: %s", e)


def main():
    logger.info("=" * 55)
    logger.info("Iniciando actualizacion de historicos oficiales...")
    summary = []

    for game, url in URLS.items():
        try:
            df_raw    = download_official_csv(url)
            df_parsed = parse_official(df_raw, game)
            inserted  = update_local_csv(game, df_parsed)
            summary.append("%s: %d filas nuevas" % (game, inserted))
        except Exception as e:
            logger.exception("[%s] Error: %s", game, e)
            summary.append("%s: ERROR - %s" % (game, e))

    report = "\n".join(summary)
    logger.info("Resumen:\n%s", report)
    logger.info("=" * 55)
    send_email_summary(report)
    logger.info("Proceso finalizado.")


if __name__ == "__main__":
    main()