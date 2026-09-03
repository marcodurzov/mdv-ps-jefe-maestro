#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advanced_stats.py
Modulo unificado de estadisticas avanzadas para Jefe Maestro v8.

Combina en un solo archivo:
  1. Correlaciones temporales de 3er orden (TC)
  2. Deteccion de sesgos fisicos y manipulacion (BIAS)
  3. Efectos por dia de sorteo (DOW)
  4. Anti-correlaciones de pares (ANTI)
  5. Analisis de rachas y ciclos (STREAK)
  6. Deteccion de cambios estructurales en el tiempo (BREAK)
  7. Analisis de entropia por bloque (ENTROPY)
  8. Correlacion resultado vs pozo acumulado (POT)

COMO FUNCIONA:
  - Se llama UNA SOLA VEZ al inicio del predictor.
  - Calcula todo y lo cachea en disco.
  - En corridas futuras carga desde cache si los datos no cambiaron.
  - Retorna un score adicional por combinacion que se suma al composite.
  - Genera un reporte de semaforo (verde/amarillo/rojo) para el correo.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(os.path.dirname(_HERE))
_CACHE    = os.path.join(_ROOT, "cache")
os.makedirs(_CACHE, exist_ok=True)

AS_CACHE_TPL  = os.path.join(_CACHE, "advanced_stats_{name}.joblib")
AS_REPORT     = os.path.join(_ROOT,  "advanced_stats_report.json")

MIN_OBS_TC       = 3
LAPLACE          = 0.5
BLOCK_SIZE       = 100
MIN_BREAK_BLOCK  = 200
SIGNIFICANCE     = 0.05

LOTTERIES = {
    "Melate":     {"n_max": 56, "k": 6, "has_bono": True},
    "Revancha":   {"n_max": 56, "k": 6, "has_bono": False},
    "Revanchita": {"n_max": 56, "k": 6, "has_bono": False},
}


def _safe(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj

def _get_combos(df: pd.DataFrame, k: int, n_max: int) -> List[List[int]]:
    num_cols = [f"N{i}" for i in range(1, k+1)]
    df_asc   = df.iloc[::-1].reset_index(drop=True)
    combos   = []
    for _, row in df_asc.iterrows():
        try:
            nums = [int(row[c]) for c in num_cols if pd.notna(row.get(c))]
            if len(nums) == k and all(1 <= n <= n_max for n in nums):
                combos.append(sorted(nums))
            else:
                combos.append([])
        except Exception:
            combos.append([])
    return combos

def _get_dates(df: pd.DataFrame) -> List[Optional[pd.Timestamp]]:
    df_asc = df.iloc[::-1].reset_index(drop=True)
    dates  = []
    for _, row in df_asc.iterrows():
        try:
            d = pd.to_datetime(row.get("FECHA",""), dayfirst=True, errors="coerce")
            dates.append(d if not pd.isna(d) else None)
        except Exception:
            dates.append(None)
    return dates


def _build_tc(combos: List[List[int]], n_max: int) -> Dict:
    T          = np.zeros((n_max, n_max, n_max), dtype=np.float32)
    pair_count = np.zeros((n_max, n_max),        dtype=np.float32)
    marginal   = np.zeros(n_max,                  dtype=np.float32)
    n_valid    = 0

    for t in range(len(combos) - 1):
        curr = combos[t]
        nxt  = combos[t+1]
        if not curr or not nxt:
            continue
        for i in range(len(curr)):
            a = curr[i] - 1
            for j in range(i+1, len(curr)):
                b = curr[j] - 1
                pair_count[a, b] += 1
                pair_count[b, a] += 1
                for c_num in nxt:
                    c = c_num - 1
                    T[a, b, c] += 1
                    T[b, a, c] += 1
        for c_num in nxt:
            marginal[c_num - 1] += 1
        n_valid += 1

    p_marg = marginal / (marginal.sum() + 1e-9)
    mask   = pair_count >= MIN_OBS_TC

    ratio = np.ones_like(T)
    for c in range(n_max):
        p_c = p_marg[c] + 1e-9
        p_c_given_ab = np.where(
            mask,
            (T[:, :, c] + LAPLACE) /
            (pair_count + LAPLACE * n_max + 1e-9),
            p_c
        )
        ratio[:, :, c] = p_c_given_ab / p_c

    return {
        "ratio":       ratio,
        "pair_count":  pair_count,
        "p_marginal":  p_marg,
        "n_valid":     n_valid,
    }


def _score_tc_batch(combos_cand: List[Tuple],
                    last_combo: List[int],
                    tc_data: Dict,
                    n_max: int) -> np.ndarray:
    ratio = tc_data["ratio"]
    pc    = tc_data["pair_count"]

    pair_ratio_vec = np.zeros(n_max, dtype=np.float64)
    n_pairs        = 0
    for i in range(len(last_combo)):
        a = last_combo[i] - 1
        for j in range(i+1, len(last_combo)):
            b = last_combo[j] - 1
            if a >= n_max or b >= n_max: continue
            if pc[a, b] < MIN_OBS_TC:   continue
            pair_ratio_vec += ratio[a, b, :]
            n_pairs        += 1

    if n_pairs == 0:
        return np.ones(len(combos_cand), dtype=np.float32)

    pair_ratio_vec /= n_pairs
    out = np.array([
        float(np.mean(pair_ratio_vec[[n-1 for n in combo if 1<=n<=n_max]]))
        for combo in combos_cand
    ], dtype=np.float32)
    return out


def _build_bias(combos: List[List[int]],
                dates:  List[Optional[pd.Timestamp]],
                n_max: int, k: int) -> Dict:
    counts   = np.zeros(n_max + 1, dtype=np.int32)
    n_sorteos = 0
    for combo in combos:
        if combo:
            for n in combo: counts[n] += 1
            n_sorteos += 1

    if n_sorteos == 0:
        return {}

    expected   = n_sorteos * k / n_max
    chi2_total = float(np.sum((counts[1:] - expected)**2 / (expected + 1e-9)))
    p_global   = float(scipy_stats.chi2.sf(chi2_total, df=n_max - 1))

    bias_per_num = {}
    for num in range(1, n_max + 1):
        obs   = counts[num]
        ratio = obs / (expected + 1e-9)
        chi2_n = (obs - expected)**2 / (expected + 1e-9)
        p_n    = float(scipy_stats.chi2.sf(chi2_n, df=1))
        bias_per_num[num] = {
            "obs":   int(obs),
            "exp":   round(expected, 2),
            "ratio": round(ratio, 3),
            "chi2":  round(chi2_n, 3),
            "p":     round(p_n, 4),
        }

    streak_max = {num: 0 for num in range(1, n_max+1)}
    streak_cur = {num: 0 for num in range(1, n_max+1)}
    for combo in combos:
        if not combo: continue
        combo_set = set(combo)
        for num in range(1, n_max+1):
            if num in combo_set:
                streak_cur[num] = 0
            else:
                streak_cur[num] += 1
                streak_max[num] = max(streak_max[num], streak_cur[num])

    p_absent    = 1 - k / n_max
    exp_streak  = 1 / (1 - p_absent + 1e-9)

    semaforo = {}
    for num in range(1, n_max+1):
        b      = bias_per_num[num]
        streak = streak_max[num]
        if b["p"] < 0.01 or streak > exp_streak * 3:
            semaforo[num] = "rojo"
        elif b["p"] < 0.05 or streak > exp_streak * 2:
            semaforo[num] = "amarillo"
        else:
            semaforo[num] = "verde"

    return {
        "chi2_global":  round(chi2_total, 3),
        "p_global":     round(p_global, 6),
        "n_sorteos":    n_sorteos,
        "expected_freq": round(expected, 2),
        "bias_per_num": bias_per_num,
        "streak_max":   streak_max,
        "exp_streak":   round(exp_streak, 1),
        "semaforo":     semaforo,
        "numeros_rojos": [n for n in range(1,n_max+1) if semaforo[n]=="rojo"],
        "numeros_amarillos": [n for n in range(1,n_max+1) if semaforo[n]=="amarillo"],
    }


def _build_dow(combos: List[List[int]],
               dates:  List[Optional[pd.Timestamp]],
               n_max: int, k: int) -> Dict:
    day_counts: Dict[int, np.ndarray] = {}
    day_totals: Dict[int, int]        = {}

    for combo, date in zip(combos, dates):
        if not combo or date is None:
            continue
        dow = date.dayofweek
        if dow not in day_counts:
            day_counts[dow] = np.zeros(n_max + 1, dtype=np.int32)
            day_totals[dow] = 0
        for n in combo:
            day_counts[dow][n] += 1
        day_totals[dow] += 1

    if len(day_counts) < 2:
        return {"disponible": False}

    global_counts = np.zeros(n_max + 1, dtype=np.int32)
    global_total  = 0
    for dc, dt in zip(day_counts.values(), day_totals.values()):
        global_counts += dc
        global_total  += dt

    dow_ratio: Dict[int, Dict[int, float]] = {}
    for dow, dc in day_counts.items():
        n_d = day_totals[dow]
        if n_d == 0: continue
        dow_ratio[dow] = {}
        for num in range(1, n_max+1):
            obs_rate    = dc[num] / (n_d * k + 1e-9)
            global_rate = global_counts[num] / (global_total * k + 1e-9)
            dow_ratio[dow][num] = round(obs_rate / (global_rate + 1e-9), 3)

    top_by_day: Dict[int, List[int]] = {}
    for dow, ratios in dow_ratio.items():
        top_by_day[dow] = sorted(
            [n for n in ratios if ratios[n] >= 1.15],
            key=lambda n: ratios[n], reverse=True
        )[:10]

    dow_names = {0:"Lunes",1:"Martes",2:"Miercoles",
                 3:"Jueves",4:"Viernes",5:"Sabado",6:"Domingo"}

    return {
        "disponible":  True,
        "dow_ratio":   {dow_names.get(d,str(d)): r for d,r in dow_ratio.items()},
        "top_by_day":  {dow_names.get(d,str(d)): v for d,v in top_by_day.items()},
        "sorteos_x_dia": {dow_names.get(d,str(d)): v
                          for d,v in day_totals.items()},
    }


def _build_anti(combos: List[List[int]], n_max: int, k: int) -> Dict:
    cooc     = np.zeros((n_max+1, n_max+1), dtype=np.int32)
    n_sorteos = 0
    for combo in combos:
        if not combo: continue
        for i in range(len(combo)):
            for j in range(i+1, len(combo)):
                a, b = combo[i], combo[j]
                cooc[a, b] += 1
                cooc[b, a] += 1
        n_sorteos += 1

    if n_sorteos == 0:
        return {}

    expected_pair = n_sorteos * k * (k-1) / (n_max * (n_max-1))

    anti_pairs   = []
    hot_pairs    = []
    for a in range(1, n_max+1):
        for b in range(a+1, n_max+1):
            obs   = cooc[a, b]
            ratio = obs / (expected_pair + 1e-9)
            if ratio < 0.3 and obs > 0:
                anti_pairs.append((a, b, round(ratio, 3), int(obs)))
            elif ratio > 2.5:
                hot_pairs.append((a, b, round(ratio, 3), int(obs)))

    anti_pairs.sort(key=lambda x: x[2])
    hot_pairs.sort(key=lambda x: x[2], reverse=True)

    return {
        "expected_pair": round(expected_pair, 2),
        "anti_pairs":    anti_pairs[:20],
        "hot_pairs":     hot_pairs[:20],
        "cooc_matrix":   cooc,
    }


def _build_structural_breaks(combos: List[List[int]],
                              n_max: int, k: int) -> Dict:
    n = len([c for c in combos if c])
    if n < MIN_BREAK_BLOCK * 2:
        return {"disponible": False, "razon": "Necesita %d+ sorteos" % (MIN_BREAK_BLOCK*2)}

    valid_combos = [c for c in combos if c]
    n_blocks     = len(valid_combos) // BLOCK_SIZE

    if n_blocks < 4:
        return {"disponible": False, "razon": "Pocos bloques"}

    block_entropy = []
    block_means   = []
    for b in range(n_blocks):
        block = valid_combos[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
        counts = np.zeros(n_max+1, dtype=np.float32)
        for combo in block:
            for n in combo: counts[n] += 1
        p   = counts[1:] / (counts[1:].sum() + 1e-9)
        ent = float(-np.sum(p * np.log2(p + 1e-12)))
        block_entropy.append(ent)
        block_means.append(float(np.mean([n for combo in block for n in combo])))

    max_ent     = float(np.log2(n_max))
    ent_arr     = np.array(block_entropy)

    min_ent_idx = int(np.argmin(ent_arr))
    min_ent_val = float(ent_arr[min_ent_idx])
    deviation   = (max_ent - min_ent_val) / max_ent * 100

    diffs      = np.abs(np.diff(ent_arr))
    break_idx  = int(np.argmax(diffs)) if len(diffs) > 0 else 0
    break_sorteo = (break_idx + 1) * BLOCK_SIZE

    return {
        "disponible":      True,
        "n_bloques":       n_blocks,
        "block_entropy":   [round(e, 4) for e in block_entropy],
        "max_entropy":     round(max_ent, 4),
        "min_entropy_bloque": min_ent_idx,
        "min_entropy_val": round(min_ent_val, 4),
        "desviacion_pct":  round(deviation, 2),
        "break_point_sorteo": break_sorteo,
        "advertencia":     deviation > 5.0,
    }


def _build_pot_correlation(combos: List[List[int]],
                            n_max: int, k: int) -> Dict:
    sums = [sum(c) for c in combos if c]
    if len(sums) < 100:
        return {"disponible": False}

    sums_arr  = np.array(sums, dtype=float)
    mean_sum  = float(np.mean(sums_arr))
    std_sum   = float(np.std(sums_arr))

    q25, q75 = float(np.percentile(sums_arr, 25)), float(np.percentile(sums_arr, 75))

    stat, p_norm = scipy_stats.normaltest(sums_arr)

    n = len(sums_arr)
    x = np.arange(n)
    slope, intercept, r, p_trend, se = scipy_stats.linregress(x, sums_arr)

    return {
        "disponible":    True,
        "mean_sum":      round(mean_sum, 2),
        "std_sum":       round(std_sum, 2),
        "q25":           round(q25, 1),
        "q75":           round(q75, 1),
        "p_normalidad":  round(float(p_norm), 4),
        "es_normal":     bool(float(p_norm) > SIGNIFICANCE),
        "tendencia_slope": round(float(slope), 4),
        "p_tendencia":   round(float(p_trend), 4),
        "hay_tendencia": bool(float(p_trend) < SIGNIFICANCE),
    }


def build_advanced_stats(df: pd.DataFrame, name: str,
                          force: bool = False) -> Dict:
    n_sorteos = len(df)
    cache_f   = AS_CACHE_TPL.format(name=name)

    if not force and os.path.exists(cache_f):
        try:
            cached = joblib.load(cache_f)
            if cached.get("n_sorteos") == n_sorteos:
                logger.info("[AS:%s] Cargado desde cache (%d sorteos)", name, n_sorteos)
                return cached
        except Exception:
            pass

    logger.info("[AS:%s] Calculando estadisticas avanzadas (%d sorteos)...", name, n_sorteos)

    n_max  = LOTTERIES[name]["n_max"]
    k      = LOTTERIES[name]["k"]
    combos = _get_combos(df, k, n_max)
    dates  = _get_dates(df)

    import time
    t0 = time.time()

    data = {"n_sorteos": n_sorteos, "name": name,
            "timestamp": datetime.now().isoformat()}

    logger.info("[AS:%s] Calculando correlaciones temporales...", name)
    try:
        data["tc"] = _build_tc(combos, n_max)
        n_v = data["tc"]["n_valid"]
        logger.info("[AS:%s] TC: %d pares de sorteos validos", name, n_v)
    except Exception as e:
        logger.warning("[AS:%s] TC fallo: %s", name, e)
        data["tc"] = {}

    logger.info("[AS:%s] Detectando sesgos fisicos...", name)
    try:
        data["bias"] = _build_bias(combos, dates, n_max, k)
        rojos = data["bias"].get("numeros_rojos", [])
        if rojos:
            logger.warning("[AS:%s] BIAS: Numeros con desviacion significativa: %s", name, rojos)
        else:
            logger.info("[AS:%s] BIAS: Sin sesgos significativos detectados", name)
    except Exception as e:
        logger.warning("[AS:%s] BIAS fallo: %s", name, e)
        data["bias"] = {}

    logger.info("[AS:%s] Analizando efecto por dia...", name)
    try:
        data["dow"] = _build_dow(combos, dates, n_max, k)
    except Exception as e:
        logger.warning("[AS:%s] DOW fallo: %s", name, e)
        data["dow"] = {"disponible": False}

    logger.info("[AS:%s] Detectando anti-correlaciones...", name)
    try:
        data["anti"] = _build_anti(combos, n_max, k)
        n_anti = len(data["anti"].get("anti_pairs", []))
        logger.info("[AS:%s] ANTI: %d pares que se evitan mucho", name, n_anti)
    except Exception as e:
        logger.warning("[AS:%s] ANTI fallo: %s", name, e)
        data["anti"] = {}

    logger.info("[AS:%s] Buscando cambios estructurales...", name)
    try:
        data["breaks"] = _build_structural_breaks(combos, n_max, k)
        if data["breaks"].get("advertencia"):
            logger.warning("[AS:%s] BREAK: Posible cambio estructural en sorteo %s",
                           name, data["breaks"].get("break_point_sorteo"))
    except Exception as e:
        logger.warning("[AS:%s] BREAKS fallo: %s", name, e)
        data["breaks"] = {"disponible": False}

    logger.info("[AS:%s] Analizando distribucion de sumas...", name)
    try:
        data["pot"] = _build_pot_correlation(combos, n_max, k)
        if data["pot"].get("hay_tendencia"):
            logger.warning("[AS:%s] POT: Tendencia temporal en sumas detectada (slope=%s)",
                           name, data['pot']['tendencia_slope'])
    except Exception as e:
        logger.warning("[AS:%s] POT fallo: %s", name, e)
        data["pot"] = {"disponible": False}

    elapsed = time.time() - t0
    logger.info("[AS:%s] Completado en %.1fs", name, elapsed)

    try:
        joblib.dump(data, cache_f)
    except Exception as e:
        logger.warning("[AS:%s] No se pudo cachear: %s", name, e)

    return data


def score_combos_advanced(combos_cand: List[Tuple[int,...]],
                           as_data: Dict,
                           last_combo: List[int],
                           n_max: int) -> np.ndarray:
    m   = len(combos_cand)
    out = np.ones(m, dtype=np.float32)

    tc = as_data.get("tc", {})
    if tc and last_combo:
        try:
            tc_scores = _score_tc_batch(combos_cand, last_combo, tc, n_max)
            out *= np.clip(tc_scores, 0.5, 3.0)
        except Exception:
            pass

    bias = as_data.get("bias", {})
    bpn  = bias.get("bias_per_num", {})
    if bpn:
        try:
            bias_vec = np.ones(n_max + 1, dtype=np.float32)
            for num, b in bpn.items():
                r = b.get("ratio", 1.0)
                if r > 1.2:
                    bias_vec[int(num)] = 1.0 + (r - 1.0) * 0.3
                elif r < 0.7:
                    bias_vec[int(num)] = 1.0 - (1.0 - r) * 0.2
            for idx, combo in enumerate(combos_cand):
                out[idx] *= float(np.mean([bias_vec[n] for n in combo
                                           if 1 <= n <= n_max]))
        except Exception:
            pass

    anti = as_data.get("anti", {})
    anti_set = set(
        (min(a,b), max(a,b))
        for a, b, r, _ in anti.get("anti_pairs", [])
        if r < 0.2
    )
    if anti_set:
        try:
            for idx, combo in enumerate(combos_cand):
                c_list = list(combo)
                n_anti = sum(1 for i in range(len(c_list))
                             for j in range(i+1, len(c_list))
                             if (min(c_list[i],c_list[j]),
                                 max(c_list[i],c_list[j])) in anti_set)
                if n_anti > 0:
                    out[idx] *= max(0.7, 1.0 - n_anti * 0.08)
        except Exception:
            pass

    return np.clip(out, 0.3, 5.0).astype(np.float32)


def generar_html_reporte(as_data_all: Dict[str, Dict]) -> str:
    if not as_data_all:
        return ""

    AZUL  = "#1a3a5c"
    html  = ("<h3 style='color:%s;border-bottom:2px solid #2e75b6;"
             "padding-bottom:6px'>Reporte de Salud Estadistica</h3>" % AZUL)
    html += ("<p style='font-size:12px;color:#666'>"
             "Analisis de sesgos fisicos, correlaciones temporales y "
             "cambios estructurales detectados en los historicos.</p>")

    for name, data in as_data_all.items():
        bias   = data.get("bias", {})
        breaks = data.get("breaks", {})
        pot    = data.get("pot",   {})
        rojos  = bias.get("numeros_rojos",     [])
        amarillos = bias.get("numeros_amarillos", [])
        p_glob = bias.get("p_global", 1.0)
        chi2   = bias.get("chi2_global", 0)

        if rojos:
            color_global = "#c62828"; icono = "ROJO"
        elif amarillos:
            color_global = "#e65100"; icono = "AMARILLO"
        else:
            color_global = "#2e7d32"; icono = "VERDE"

        tend_txt = ""
        if pot.get("hay_tendencia"):
            s = pot.get("tendencia_slope", 0)
            tend_txt = ("Tendencia en sumas: %s %.3f/sorteo<br>" %
                        ("subiendo" if s > 0 else "bajando", abs(s)))

        break_txt = ""
        if breaks.get("advertencia"):
            break_txt = ("Posible cambio estructural en sorteo #%s (%s%% desviacion)<br>" %
                         (breaks.get('break_point_sorteo','?'),
                          breaks.get('desviacion_pct','?')))

        rojos_str    = " ".join(str(n) for n in rojos[:10])    if rojos    else "Ninguno"
        amarillos_str = " ".join(str(n) for n in amarillos[:10]) if amarillos else "Ninguno"

        html += (
            "<div style='background:#f9f9f9;border-left:4px solid %s;"
            "border-radius:4px;padding:10px;margin-bottom:10px'>"
            "<b style='color:%s'>%s</b> &nbsp; %s &nbsp;"
            "<span style='font-size:12px;color:#888'>chi2=%.1f | p=%.4f</span><br>"
            "<span style='font-size:12px'>"
            "Desviacion significativa: <b>%s</b><br>"
            "Desviacion leve: <b>%s</b><br>"
            "%s%s"
            "</span></div>"
        ) % (color_global, AZUL, name, icono, chi2, p_glob,
             rojos_str, amarillos_str, tend_txt, break_txt)

    html += ("<p style='font-size:10px;color:#aaa'>"
             "ROJO: p&lt;0.01 o racha &gt;3x esperada | "
             "AMARILLO: p&lt;0.05 o racha &gt;2x esperada | "
             "VERDE: distribucion normal. "
             "Nota: desviaciones pueden ser varianza natural con N&lt;15,000.</p>")
    return html


def load_or_build_all(all_histories: Dict[str, pd.DataFrame],
                       force: bool = False) -> Dict[str, Dict]:
    result = {}
    for name, df in all_histories.items():
        try:
            result[name] = build_advanced_stats(df, name, force=force)
        except Exception as e:
            logger.error("[AS:%s] Error fatal: %s", name, e)
            result[name] = {"n_sorteos": len(df), "name": name}

    try:
        report = {}
        for name, data in result.items():
            bias   = data.get("bias", {})
            breaks = data.get("breaks", {})
            report[name] = {
                "p_global":         bias.get("p_global", 1.0),
                "numeros_rojos":    bias.get("numeros_rojos", []),
                "numeros_amarillos":bias.get("numeros_amarillos", []),
                "break_detectado":  breaks.get("advertencia", False),
                "timestamp":        data.get("timestamp", ""),
            }
        with open(AS_REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=_safe)
    except Exception:
        pass

    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    _ROOT_T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DATA_T = os.path.join(_ROOT_T, "data")

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
            print("[%s] %d sorteos" % (game, len(df)))

    if histories:
        result = load_or_build_all(histories, force=True)
        for name, data in result.items():
            print("=" * 50)
            print("[%s]" % name)
            bias = data.get("bias", {})
            print("  Chi2 global: %s (p=%s)" % (bias.get('chi2_global','N/A'), bias.get('p_global','N/A')))
            print("  Numeros rojos: %s" % bias.get('numeros_rojos', []))
    else:
        print("No se encontraron CSVs.")