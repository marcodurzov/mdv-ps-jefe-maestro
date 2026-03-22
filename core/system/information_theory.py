#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
information_theory.py
Modulo unificado de teoria de la informacion, termodinamica y caos
aplicados al analisis de Melate, Revancha y Revanchita.

CONTENIDO:

1. INFORMACION MUTUA NO LINEAL
   Mientras la correlacion de Pearson mide relaciones LINEALES entre
   numeros, la informacion mutua mide CUALQUIER tipo de dependencia.
   I(X;Y) = H(X) + H(Y) - H(X,Y)
   donde H es la entropia de Shannon.
   Un par con I > 0 tiene alguna forma de dependencia, lineal o no.

2. PERSPECTIVA TERMODINAMICA
   El bombo es un sistema termodinámico. La temperatura y estacion del
   año afectan la densidad del aire y la friccion, lo que puede generar
   sesgos estacionales detectables con suficientes datos.
   Analiza si la distribucion de numeros cambia por trimestre del año.

3. TEORIA DEL CAOS - EXPONENTE DE LYAPUNOV
   Un sistema caotico tiene un exponente de Lyapunov positivo.
   Si el bombo tiene dinamica caotica determinista, la serie temporal
   de sumas deberia mostrar un exponente de Lyapunov calculable.
   Esto nos dice si hay estructura no aleatoria explotable.

INTEGRACION:
   Se llama desde jefe_maestro_v6_elite_predictor.py.
   Retorna scores adicionales por combinacion y HTML para el correo.
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

IT_CACHE_TPL = os.path.join(_CACHE, "info_theory_{name}.joblib")


# ─────────────────────────────────────────────────────────────────────
# 1. INFORMACION MUTUA NO LINEAL
# ─────────────────────────────────────────────────────────────────────

def _entropy_1d(x, n_bins=10):
    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts / (counts.sum() + 1e-9)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _entropy_2d(x, y, n_bins=10):
    counts, _, _ = np.histogram2d(x, y, bins=n_bins)
    probs = counts / (counts.sum() + 1e-9)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _mutual_information(x, y, n_bins=10):
    hx  = _entropy_1d(x, n_bins)
    hy  = _entropy_1d(y, n_bins)
    hxy = _entropy_2d(x, y, n_bins)
    return max(0.0, hx + hy - hxy)


def build_mutual_information(combos, n_max, n_bins=8):
    """
    Calcula la informacion mutua entre cada par de numeros
    basandose en su co-aparicion a lo largo del tiempo.
    """
    n = len(combos)
    if n < 50:
        return np.ones((n_max+1, n_max+1), dtype=np.float32)

    # Serie temporal de aparicion por numero (0/1 por sorteo)
    presence = np.zeros((n_max+1, n), dtype=np.float32)
    for t, combo in enumerate(combos):
        if combo:
            for num in combo:
                if 1 <= num <= n_max:
                    presence[num, t] = 1.0

    # Calcular MI para cada par
    mi_matrix = np.zeros((n_max+1, n_max+1), dtype=np.float32)
    for a in range(1, n_max+1):
        for b in range(a+1, n_max+1):
            mi = _mutual_information(presence[a], presence[b], n_bins)
            mi_matrix[a, b] = mi
            mi_matrix[b, a] = mi

    return mi_matrix


def score_mi_batch(combos_cand, mi_matrix, n_max):
    """
    Score de informacion mutua para candidatos.
    Combos cuyos numeros tienen alta MI entre si (aparecen
    de forma estadisticamente no independiente) reciben mayor score.
    """
    m = len(combos_cand)
    scores = np.zeros(m, dtype=np.float32)
    for i, combo in enumerate(combos_cand):
        pair_mi = []
        for j in range(len(combo)):
            for k in range(j+1, len(combo)):
                a, b = int(combo[j]), int(combo[k])
                if 1 <= a <= n_max and 1 <= b <= n_max:
                    pair_mi.append(float(mi_matrix[a, b]))
        scores[i] = float(np.mean(pair_mi)) if pair_mi else 0.0

    # Normalizar a rango 0.85 - 1.15
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (0.85 + 0.30 * (scores - s_min) / (s_max - s_min)).astype(np.float32)
    return np.ones(m, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# 2. PERSPECTIVA TERMODINAMICA (estaciones del año)
# ─────────────────────────────────────────────────────────────────────

def build_seasonal_bias(df, n_max, k):
    """
    Analiza si la distribucion de numeros cambia por trimestre.
    Trimestres: Q1=Ene-Mar, Q2=Abr-Jun, Q3=Jul-Sep, Q4=Oct-Dic
    Retorna el trimestre actual y los numeros favorecidos en el.
    """
    ncols = ["N%d" % i for i in range(1, k+1)]

    if "FECHA" not in df.columns:
        return {"disponible": False}

    df_work = df.copy()
    df_work["_fecha_dt"] = pd.to_datetime(
        df_work["FECHA"], dayfirst=True, errors="coerce"
    )
    df_work = df_work.dropna(subset=["_fecha_dt"])
    df_work["_quarter"] = df_work["_fecha_dt"].dt.quarter
    df_work["_month"]   = df_work["_fecha_dt"].dt.month

    # Contar apariciones por numero y trimestre
    q_counts = {q: np.zeros(n_max+1, dtype=np.float32) for q in range(1, 5)}
    q_totals = {q: 0 for q in range(1, 5)}

    for _, row in df_work.iterrows():
        q = int(row["_quarter"])
        for col in ncols:
            if pd.notna(row.get(col)):
                v = int(row[col])
                if 1 <= v <= n_max:
                    q_counts[q][v] += 1
        q_totals[q] += 1

    # Ratio por trimestre vs global
    global_counts = sum(q_counts.values())
    global_total  = sum(q_totals.values())

    seasonal_ratio = {}
    for q in range(1, 5):
        if q_totals[q] == 0:
            continue
        ratio = np.ones(n_max+1, dtype=np.float32)
        for num in range(1, n_max+1):
            obs_rate    = q_counts[q][num] / (q_totals[q] * k + 1e-9)
            global_rate = global_counts[num] / (global_total * k + 1e-9)
            ratio[num]  = float(obs_rate / (global_rate + 1e-9))
        seasonal_ratio[q] = ratio

    # Trimestre actual
    current_quarter = pd.Timestamp.now().quarter
    current_month   = pd.Timestamp.now().month

    # Numeros favorecidos en el trimestre actual
    if current_quarter in seasonal_ratio:
        ratios = seasonal_ratio[current_quarter]
        favorecidos = sorted(
            [(num, float(ratios[num])) for num in range(1, n_max+1)],
            key=lambda x: x[1], reverse=True
        )[:10]
        penalizados = sorted(
            [(num, float(ratios[num])) for num in range(1, n_max+1)],
            key=lambda x: x[1]
        )[:10]
    else:
        favorecidos = []
        penalizados = []

    # Nombre del trimestre
    quarter_names = {1: "Enero-Marzo", 2: "Abril-Junio",
                     3: "Julio-Septiembre", 4: "Octubre-Diciembre"}

    return {
        "disponible":       True,
        "trimestre_actual": current_quarter,
        "mes_actual":       current_month,
        "nombre_trimestre": quarter_names.get(current_quarter, ""),
        "seasonal_ratio":   {str(q): r.tolist() for q, r in seasonal_ratio.items()},
        "favorecidos":      favorecidos,
        "penalizados":      penalizados,
        "sorteos_por_q":    q_totals,
    }


def score_seasonal_batch(combos_cand, seasonal_data, n_max):
    """
    Score basado en sesgo estacional.
    Favorece numeros que historicamente han aparecido mas
    en el trimestre actual.
    """
    m = len(combos_cand)
    if not seasonal_data.get("disponible"):
        return np.ones(m, dtype=np.float32)

    q  = str(seasonal_data.get("trimestre_actual", 1))
    sr = seasonal_data.get("seasonal_ratio", {})
    if q not in sr:
        return np.ones(m, dtype=np.float32)

    ratio_vec = np.array(sr[q], dtype=np.float32)

    scores = np.array([
        float(np.mean([ratio_vec[n] for n in combo if 1 <= n <= n_max]))
        for combo in combos_cand
    ], dtype=np.float32)

    # Normalizar a 0.90 - 1.10
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        return (0.90 + 0.20 * (scores - s_min) / (s_max - s_min)).astype(np.float32)
    return np.ones(m, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
# 3. TEORIA DEL CAOS - EXPONENTE DE LYAPUNOV
# ─────────────────────────────────────────────────────────────────────

def estimate_lyapunov(series, lag=1, embedding_dim=3):
    """
    Estima el exponente de Lyapunov maximo usando el algoritmo de Rosenstein.
    Un exponente positivo indica dinamica caotica.
    Un exponente cercano a 0 indica comportamiento mas predecible.

    Args:
        series: Serie temporal 1D (ej: sumas de combinaciones ganadoras)
        lag:    Retardo de embedding
        embedding_dim: Dimension del espacio de fases reconstruido
    """
    n = len(series)
    if n < 50:
        return {"disponible": False, "lyapunov": None}

    # Normalizar
    series = (series - series.mean()) / (series.std() + 1e-9)

    # Reconstruccion del espacio de fases (teorema de Takens)
    m = n - (embedding_dim - 1) * lag
    if m < 20:
        return {"disponible": False, "lyapunov": None}

    embedded = np.zeros((m, embedding_dim), dtype=np.float64)
    for i in range(m):
        for d in range(embedding_dim):
            embedded[i, d] = series[i + d * lag]

    # Para cada punto, encontrar el vecino mas cercano
    # que no sea vecino temporal (separacion minima = 10 pasos)
    min_temporal_sep = max(10, n // 20)
    divergences = []

    for i in range(m - min_temporal_sep):
        # Distancias al punto i
        diffs    = embedded - embedded[i]
        dists    = np.sqrt(np.sum(diffs**2, axis=1))
        # Excluir vecinos temporales
        dists[:max(0, i - min_temporal_sep)] = np.inf
        dists[i:min(m, i + min_temporal_sep)] = np.inf

        if np.all(np.isinf(dists)):
            continue

        j = np.argmin(dists)
        d0 = dists[j]
        if d0 < 1e-10:
            continue

        # Seguir la divergencia
        steps_ahead = min(10, m - max(i, j) - 1)
        if steps_ahead < 2:
            continue

        log_divs = []
        for t in range(1, steps_ahead):
            if i + t < m and j + t < m:
                dt = np.sqrt(np.sum((embedded[i+t] - embedded[j+t])**2))
                if dt > 1e-10:
                    log_divs.append(np.log(dt / d0))

        if log_divs:
            divergences.append(np.mean(log_divs))

    if not divergences:
        return {"disponible": False, "lyapunov": None}

    lyapunov = float(np.mean(divergences))

    # Interpretacion
    if lyapunov > 0.1:
        interpretacion = "Sistema caotico - alta sensibilidad a condiciones iniciales"
        predecibilidad = "Baja"
    elif lyapunov > 0:
        interpretacion = "Sistema levemente caotico - alguna estructura explotable"
        predecibilidad = "Media"
    else:
        interpretacion = "Sistema no caotico - comportamiento mas regular"
        predecibilidad = "Alta"

    return {
        "disponible":      True,
        "lyapunov":        round(lyapunov, 4),
        "interpretacion":  interpretacion,
        "predecibilidad":  predecibilidad,
        "n_puntos":        len(divergences),
    }


def build_chaos_analysis(combos, n_max, k):
    """
    Analisis completo de caos para la serie temporal de sorteos.
    """
    valid = [c for c in combos if c]
    if len(valid) < 50:
        return {"disponible": False}

    # Serie de sumas (proxy del estado del sistema)
    sums_series = np.array([sum(c) for c in valid], dtype=np.float64)

    # Serie de rango (max - min de cada sorteo)
    range_series = np.array([max(c) - min(c) for c in valid], dtype=np.float64)

    # Serie de entropia por sorteo
    entropy_series = []
    for c in valid:
        bins  = np.histogram(c, bins=6, range=(1, n_max+1))[0]
        p     = bins / (bins.sum() + 1e-9)
        ent   = float(-np.sum(p * np.log2(p + 1e-12)))
        entropy_series.append(ent)
    entropy_series = np.array(entropy_series, dtype=np.float64)

    # Exponente de Lyapunov en serie de sumas
    lyap_sums    = estimate_lyapunov(sums_series)
    lyap_entropy = estimate_lyapunov(entropy_series)

    # Estadisticas de las series
    return {
        "disponible":     True,
        "n_sorteos":      len(valid),
        "lyapunov_sums":  lyap_sums,
        "lyapunov_entropy": lyap_entropy,
        "sums_mean":      round(float(sums_series.mean()), 2),
        "sums_std":       round(float(sums_series.std()), 2),
        "range_mean":     round(float(range_series.mean()), 2),
        "entropy_mean":   round(float(entropy_series.mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────
# CONSTRUCTOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

def build_information_theory(df, name, force=False):
    """
    Construye todos los analisis de teoria de la informacion.
    Usa cache en disco si los datos no cambiaron.
    """
    from itertools import islice

    n = len(df)
    cache_f = IT_CACHE_TPL.format(name=name)

    if not force and os.path.exists(cache_f):
        try:
            cached = joblib.load(cache_f)
            if cached.get("n_sorteos") == n:
                logger.info("[IT:%s] Cargado desde cache (%d sorteos)", name, n)
                return cached
        except Exception:
            pass

    logger.info("[IT:%s] Calculando teoria de la informacion (%d sorteos)...", name, n)

    import time
    t0 = time.time()

    n_max = 56
    k     = 6
    ncols = ["N%d" % i for i in range(1, k+1)]

    # Extraer combinaciones en orden ascendente
    df_asc = df.iloc[::-1].reset_index(drop=True)
    combos = []
    for _, row in df_asc.iterrows():
        try:
            nums = [int(row[c]) for c in ncols
                    if pd.notna(row.get(c)) and 1 <= int(row[c]) <= n_max]
            if len(nums) == k:
                combos.append(sorted(nums))
            else:
                combos.append([])
        except Exception:
            combos.append([])

    data = {"n_sorteos": n, "name": name}

    # 1. Informacion mutua
    logger.info("[IT:%s] Calculando informacion mutua...", name)
    try:
        mi_matrix = build_mutual_information(combos, n_max)
        # Guardar solo los pares con MI significativa para ahorrar espacio
        mi_pairs = {}
        threshold = float(np.percentile(
            [mi_matrix[a,b] for a in range(1,n_max+1) for b in range(a+1,n_max+1)],
            75
        ))
        for a in range(1, n_max+1):
            for b in range(a+1, n_max+1):
                v = float(mi_matrix[a, b])
                if v >= threshold:
                    mi_pairs["%d_%d" % (a, b)] = round(v, 4)

        data["mi_matrix_full"]  = mi_matrix  # array completo para scoring
        data["mi_pairs"]        = mi_pairs   # solo significativos para reporte
        data["mi_threshold"]    = round(threshold, 4)
        logger.info("[IT:%s] MI: %d pares significativos", name, len(mi_pairs))
    except Exception as e:
        logger.warning("[IT:%s] MI fallo: %s", name, e)
        data["mi_matrix_full"] = None
        data["mi_pairs"]       = {}

    # 2. Analisis estacional
    logger.info("[IT:%s] Calculando sesgo estacional...", name)
    try:
        seasonal = build_seasonal_bias(df, n_max, k)
        data["seasonal"] = seasonal
        if seasonal.get("disponible"):
            fav = seasonal.get("favorecidos", [])[:3]
            logger.info("[IT:%s] Estacional Q%d favorecidos: %s",
                        name, seasonal.get("trimestre_actual", "?"),
                        [f[0] for f in fav])
    except Exception as e:
        logger.warning("[IT:%s] Estacional fallo: %s", name, e)
        data["seasonal"] = {"disponible": False}

    # 3. Analisis de caos
    logger.info("[IT:%s] Calculando exponente de Lyapunov...", name)
    try:
        chaos = build_chaos_analysis(combos, n_max, k)
        data["chaos"] = chaos
        if chaos.get("disponible"):
            lyap = chaos.get("lyapunov_sums", {}).get("lyapunov")
            logger.info("[IT:%s] Lyapunov sums: %s", name, lyap)
    except Exception as e:
        logger.warning("[IT:%s] Caos fallo: %s", name, e)
        data["chaos"] = {"disponible": False}

    elapsed = time.time() - t0
    logger.info("[IT:%s] Completado en %.1fs", name, elapsed)

    try:
        joblib.dump(data, cache_f)
    except Exception as e:
        logger.warning("[IT:%s] No se pudo cachear: %s", name, e)

    return data


def load_or_build_it(all_histories, force=False):
    result = {}
    for name, df in all_histories.items():
        try:
            result[name] = build_information_theory(df, name, force=force)
        except Exception as e:
            logger.error("[IT:%s] Error fatal: %s", name, e)
            result[name] = {"n_sorteos": len(df), "name": name}
    return result


# ─────────────────────────────────────────────────────────────────────
# SCORING UNIFICADO
# ─────────────────────────────────────────────────────────────────────

def score_it_batch(combos_cand, it_data, n_max=56):
    """
    Score combinado de MI + estacional para un batch de candidatos.
    """
    m = len(combos_cand)
    score = np.ones(m, dtype=np.float32)

    # Informacion mutua
    mi_matrix = it_data.get("mi_matrix_full")
    if mi_matrix is not None:
        try:
            mi_scores = score_mi_batch(combos_cand, mi_matrix, n_max)
            score = score * mi_scores
        except Exception:
            pass

    # Estacional
    seasonal = it_data.get("seasonal", {})
    if seasonal.get("disponible"):
        try:
            sea_scores = score_seasonal_batch(combos_cand, seasonal, n_max)
            score = score * sea_scores
        except Exception:
            pass

    return np.clip(score, 0.5, 2.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# HTML PARA EL CORREO
# ─────────────────────────────────────────────────────────────────────

def generar_html_it(it_data_all):
    if not it_data_all:
        return ""

    AZUL = "#1a3a5c"
    html = (
        "<h3 style='color:%s;border-bottom:2px solid #2e75b6;"
        "padding-bottom:6px'>Analisis de Informacion y Caos</h3>" % AZUL
    )

    for name, data in it_data_all.items():
        seasonal = data.get("seasonal", {})
        chaos    = data.get("chaos", {})
        mi_pairs = data.get("mi_pairs", {})

        # Estacional
        if seasonal.get("disponible"):
            q_name = seasonal.get("nombre_trimestre", "")
            fav    = seasonal.get("favorecidos", [])[:5]
            fav_str = ", ".join(str(f[0]) for f in fav)
        else:
            q_name  = "No disponible"
            fav_str = "N/A"

        # Caos - Lyapunov
        lyap_info  = chaos.get("lyapunov_sums", {})
        lyap_val   = lyap_info.get("lyapunov")
        lyap_interp = lyap_info.get("interpretacion", "No calculado")

        if lyap_val is not None:
            lyap_color = "#c62828" if lyap_val > 0.1 else (
                "#e65100" if lyap_val > 0 else "#2e7d32"
            )
            lyap_str = "%.4f (%s)" % (lyap_val, lyap_info.get("predecibilidad",""))
        else:
            lyap_color = "#666"
            lyap_str   = "Insuficientes datos"

        # MI pares mas significativos
        top_mi = sorted(mi_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        mi_str = ", ".join("(%s)" % k.replace("_","-") for k,_ in top_mi) if top_mi else "N/A"

        html += (
            "<div style='background:#f9f9f9;border-radius:4px;"
            "padding:10px;margin-bottom:10px;border-left:4px solid %s'>"
            "<b style='color:%s'>%s</b><br>"
            "<span style='font-size:12px'>"
            "<b>Trimestre %s:</b> Numeros favorecidos historicamente: %s<br>"
            "<b>Exponente de Lyapunov:</b> "
            "<span style='color:%s'>%s</span><br>"
            "<span style='font-size:11px;color:#888'>%s</span><br>"
            "<b>Pares con mayor informacion mutua:</b> %s"
            "</span></div>"
        ) % (
            AZUL, AZUL, name,
            q_name, fav_str,
            lyap_color, lyap_str,
            lyap_interp,
            mi_str
        )

    html += (
        "<p style='font-size:10px;color:#aaa'>"
        "Informacion mutua: dependencias no lineales entre numeros. "
        "Lyapunov: sensibilidad del sistema a perturbaciones (>0 = caotico). "
        "Estacional: sesgos historicos por trimestre del año."
        "</p>"
    )
    return html


if __name__ == "__main__":
    import sys
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s - %(levelname)s - %(message)s")

    _DATA = os.path.join(_ROOT, "data")
    histories = {}
    for game in ["Melate", "Revancha", "Revanchita"]:
        path = os.path.join(_DATA, "%s.csv" % game.lower())
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["_fecha_dt"] = pd.to_datetime(
                df["FECHA"], dayfirst=True, errors="coerce"
            )
            df = df.sort_values("_fecha_dt", ascending=False).reset_index(drop=True)
            histories[game] = df
            print("[%s] %d sorteos" % (game, len(df)))

    if histories:
        result = load_or_build_it(histories, force=True)
        for name, data in result.items():
            print("\n[%s] Chaos: %s" % (name, data.get("chaos", {}).get("lyapunov_sums")))
            print("[%s] Seasonal Q%s favorecidos: %s" % (
                name,
                data.get("seasonal", {}).get("trimestre_actual","?"),
                [f[0] for f in data.get("seasonal", {}).get("favorecidos", [])[:5]]
            ))