#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
social_bias.py
Perspectiva sociologica del jugador para Jefe Maestro v8.

CONCEPTO CENTRAL:
  Si ganas el jackpot pero lo compartes con 50 personas, tu ganancia
  real es 1/50 del pozo. La estrategia correcta no es solo maximizar
  probabilidad de acertar, sino maximizar el valor esperado real.

Este modulo:
  1. Modela que combinaciones elige la gente (para evitarlas)
  2. Usa la columna BOLSA para detectar patrones de comportamiento
     colectivo y estimar popularidad relativa de combinaciones
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(os.path.dirname(_HERE))
_CACHE = os.path.join(_ROOT, "cache")
os.makedirs(_CACHE, exist_ok=True)

SB_CACHE_TPL = os.path.join(_CACHE, "social_bias_{name}.joblib")


def _popularity_score(combo: tuple, n_max: int = 56) -> float:
    combo = sorted(combo)
    k = len(combo)
    score = 0.0

    n_low = sum(1 for n in combo if n <= 31)
    low_ratio = n_low / k
    score += low_ratio * 0.25

    n_month = sum(1 for n in combo if n <= 12)
    if n_low >= 4 and n_month >= 2:
        score += 0.30

    for step in [5, 7, 10]:
        n_seq = sum(1 for n in combo if n % step == 0)
        if n_seq >= 3:
            score += 0.10

    consec = sum(1 for i in range(k-1) if combo[i+1] == combo[i] + 1)
    if consec >= 2:
        score += 0.10

    n_even = sum(1 for n in combo if n % 2 == 0)
    if n_even == k or n_even == 0:
        score += 0.15

    decenas = [n // 10 for n in combo]
    max_same_decena = max(decenas.count(d) for d in set(decenas))
    if max_same_decena >= 4:
        score += 0.20

    lucky = {3, 7, 13, 21, 33}
    n_lucky = sum(1 for n in combo if n in lucky)
    score += n_lucky * 0.04

    s = sum(combo)
    if s % 50 == 0:
        score += 0.10
    elif s % 25 == 0:
        score += 0.05

    diagonal = {1, 9, 17, 25, 33, 41, 49}
    n_diag = sum(1 for n in combo if n in diagonal)
    if n_diag >= 4:
        score += 0.15

    return min(score, 1.0)


def _unpopularity_score(combo: tuple, n_max: int = 56) -> float:
    return 1.0 - _popularity_score(combo, n_max)


def analyze_bolsa(df: pd.DataFrame, name: str) -> Dict:
    if "BOLSA" not in df.columns:
        return {"disponible": False}

    df_asc = df.iloc[::-1].reset_index(drop=True).copy()
    df_asc["BOLSA"] = pd.to_numeric(df_asc["BOLSA"], errors="coerce")
    df_asc = df_asc.dropna(subset=["BOLSA"])

    if len(df_asc) < 10:
        return {"disponible": False}

    bolsas = df_asc["BOLSA"].values.astype(float)

    resets = []
    for i in range(1, len(bolsas)):
        if bolsas[i] < bolsas[i-1] * 0.5:
            resets.append(i)

    increments = np.diff(bolsas)
    increments_pos = increments[increments > 0]

    ticket_price = 15.0
    pozo_pct = 0.22
    tickets_per_draw = increments_pos / (ticket_price * pozo_pct)

    p25 = float(np.percentile(bolsas, 25))
    p50 = float(np.percentile(bolsas, 50))
    p75 = float(np.percentile(bolsas, 75))
    current = float(bolsas[-1]) if len(bolsas) > 0 else 0.0

    if p75 > p25:
        pot_level = (current - p25) / (p75 - p25)
    else:
        pot_level = 0.5
    pot_level = float(np.clip(pot_level, 0.0, 2.0))

    last_reset = resets[-1] if resets else 0
    sorteos_sin_ganador = len(bolsas) - 1 - last_reset

    return {
        "disponible":           True,
        "n_jackpots_detectados": len(resets),
        "sorteos_sin_ganador":  sorteos_sin_ganador,
        "pozo_actual":          current,
        "pozo_p25":             p25,
        "pozo_p50":             p50,
        "pozo_p75":             p75,
        "pot_level":            round(pot_level, 3),
        "tickets_promedio_est": round(float(np.mean(tickets_per_draw)), 0) if len(tickets_per_draw) > 0 else 0,
        "tickets_max_est":      round(float(np.max(tickets_per_draw)), 0) if len(tickets_per_draw) > 0 else 0,
    }


def score_social_bias_batch(
    combos: List[Tuple[int, ...]],
    bolsa_data: Dict,
    n_max: int = 56
) -> np.ndarray:
    m = len(combos)
    scores = np.zeros(m, dtype=np.float32)

    for i, combo in enumerate(combos):
        scores[i] = _unpopularity_score(combo, n_max)

    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores_norm = 0.7 + 0.6 * (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = np.ones(m, dtype=np.float32)

    pot_level = bolsa_data.get("pot_level", 1.0) if bolsa_data.get("disponible") else 1.0
    amplifier = 1.0 + min(pot_level - 1.0, 0.5) * 0.3

    return (scores_norm * amplifier).astype(np.float32)


def generar_html_social(bolsa_data_all: Dict[str, Dict]) -> str:
    if not bolsa_data_all:
        return ""

    AZUL = "#1a3a5c"
    html = (
        "<h3 style='color:%s;border-bottom:2px solid #2e75b6;"
        "padding-bottom:6px'>Analisis de Pozo Acumulado</h3>" % AZUL
    )

    for name, data in bolsa_data_all.items():
        if not data.get("disponible"):
            continue

        pozo    = data.get("pozo_actual", 0)
        p50     = data.get("pozo_p50", 0)
        level   = data.get("pot_level", 1.0)
        sin_gan = data.get("sorteos_sin_ganador", 0)
        tickets = data.get("tickets_promedio_est", 0)

        if level > 1.5:
            color = "#c62828"; icono = "Pozo muy alto"
        elif level > 1.0:
            color = "#e65100"; icono = "Pozo alto"
        else:
            color = "#2e7d32"; icono = "Pozo normal"

        pozo_str = "${:,.0f}".format(pozo)
        p50_str  = "${:,.0f}".format(p50)

        html += (
            "<div style='background:#f9f9f9;border-left:4px solid %s;"
            "border-radius:4px;padding:10px;margin-bottom:10px'>"
            "<b style='color:%s'>%s</b> &nbsp;%s<br>"
            "<span style='font-size:13px'>"
            "Pozo actual: <b>%s MXN</b> (mediana historica: %s MXN)<br>"
            "Sorteos sin ganador mayor: <b>%d</b><br>"
            "Tickets estimados por sorteo: ~%s<br>"
            "</span>"
            "<p style='font-size:11px;color:#888;margin:4px 0'>"
            "Estrategia: con pozo alto hay mas jugadores. "
            "Las combinaciones impopulares tienen mayor valor esperado "
            "porque si ganas, compartes menos el premio."
            "</p></div>"
        ) % (color, AZUL, name, icono, pozo_str, p50_str,
             sin_gan, "{:,.0f}".format(tickets))

    html += (
        "<p style='font-size:10px;color:#aaa'>"
        "Estimacion de tickets basada en incremento del pozo. "
        "Las combinaciones del Top 20 ya fueron filtradas para "
        "maximizar impopularidad relativa."
        "</p>"
    )
    return html


def build_social_bias(df: pd.DataFrame, name: str,
                       force: bool = False) -> Dict:
    n = len(df)
    cache_f = SB_CACHE_TPL.format(name=name)

    if not force and os.path.exists(cache_f):
        try:
            cached = joblib.load(cache_f)
            if cached.get("n_sorteos") == n:
                logger.info("[SB:%s] Cargado desde cache", name)
                return cached
        except Exception:
            pass

    logger.info("[SB:%s] Calculando sesgo social (%d sorteos)...", name, n)

    bolsa_data = analyze_bolsa(df, name)

    data = {
        "n_sorteos":  n,
        "name":       name,
        "bolsa":      bolsa_data,
    }

    if bolsa_data.get("disponible"):
        logger.info(
            "[SB:%s] Pozo actual: $%s | Level: %.2f | Sin ganador: %d sorteos",
            name,
            "{:,.0f}".format(bolsa_data.get("pozo_actual", 0)),
            bolsa_data.get("pot_level", 1.0),
            bolsa_data.get("sorteos_sin_ganador", 0)
        )

    try:
        joblib.dump(data, cache_f)
    except Exception as e:
        logger.warning("[SB:%s] No se pudo cachear: %s", name, e)

    return data


def load_or_build_social(all_histories: Dict,
                          force: bool = False) -> Dict[str, Dict]:
    result = {}
    for name, df in all_histories.items():
        try:
            result[name] = build_social_bias(df, name, force=force)
        except Exception as e:
            logger.error("[SB:%s] Error: %s", name, e)
            result[name] = {"n_sorteos": len(df), "name": name,
                            "bolsa": {"disponible": False}}
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    _DATA_T = os.path.join(_ROOT, "data")
    histories = {}
    for game in ["Melate", "Revancha", "Revanchita"]:
        path = os.path.join(_DATA_T, "%s.csv" % game.lower())
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["_fecha_dt"] = pd.to_datetime(
                df["FECHA"], dayfirst=True, errors="coerce"
            )
            df = df.sort_values("_fecha_dt", ascending=False).reset_index(drop=True)
            histories[game] = df
            print("[%s] %d sorteos cargados" % (game, len(df)))

    if histories:
        result = load_or_build_social(histories, force=True)
        for name, data in result.items():
            print("[%s] Bolsa: %s" % (name, data.get("bolsa", {})))
    else:
        print("No se encontraron CSVs.")