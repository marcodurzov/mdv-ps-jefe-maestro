#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retroactive_migration.py
Script de UNA SOLA EJECUCION para poblar retroactive_tracking.json
con historial retroactivo simulado de los ultimos N sorteos reales.

POR QUE ES NECESARIO:
  predictions_history_v8.json solo existe desde que el sistema actual
  arranco. No hay predicciones ML guardadas de meses/anios atras porque
  el modelo de hoy no existia entonces. Este script resuelve eso
  simulando, para cada sorteo historico, que hubiera predicho un
  sistema basado en estadisticas (hot/gap/co-ocurrencia) usando SOLO
  la informacion disponible ANTES de ese sorteo (walk-forward, sin
  fuga de datos hacia el futuro).

POR QUE NO USA LOS MODELOS ML COMPLETOS (XGB/LGBM/CatBoost/LSTM):
  Entrenar el ensemble completo tarda ~25-30 min POR JUEGO en cada
  corrida. Repetirlo 150 veces (una por sorteo historico) tardaria
  dias y no es viable en GitHub Actions. En su lugar se usa el mismo
  motor estadistico rapido que ya usa la funcion backtest() del
  predictor principal (hot30 + gap + co-ocurrencia), que es la
  aproximacion mas fiel posible sin re-entrenar ML en cada paso.
  Esto es una aproximacion honesta, no un reemplazo del pipeline ML.

QUE HACE EXACTAMENTE:
  1. Por cada loteria (Melate, Revancha, Revanchita):
     a. Toma los ultimos MIGRATION_DEPTH sorteos reales (mas recientes)
     b. Para cada uno de esos sorteos, "retrocede en el tiempo":
        usa SOLO los sorteos anteriores a esa fecha para calcular
        hot30/gap/co-ocurrencia (sin ver el futuro)
     c. Genera un top-20 equivalente ordenado por ese score estadistico
     d. Compara contra el resultado real de ese sorteo
     e. Guarda la entrada en el mismo formato que usa retroactive_learner.py
  2. Escribe (o AMPLIA si ya existe) retroactive_tracking.json
  3. Marca cada entrada con "source": "migration" para diferenciarlas
     de las entradas generadas en tiempo real por el sistema normal

SE EJECUTA UNA SOLA VEZ. Despues de esta migracion, el flujo normal
(main_run.py -> retroactive_learner.py) continua solo, sorteo a sorteo,
sin necesidad de volver a correr este script.

USO:
  python core/system/retroactive_migration.py
  python core/system/retroactive_migration.py --depth 200
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("migration")

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(os.path.dirname(_HERE))
_DATA_DIR = os.path.join(_ROOT, "data")

RETROACTIVE_FILE = os.path.join(_ROOT, "retroactive_tracking.json")

LOTTERIES = {
    "Melate":     {"n_max": 56, "k": 6, "has_bono": True},
    "Revancha":   {"n_max": 56, "k": 6, "has_bono": False},
    "Revanchita": {"n_max": 56, "k": 6, "has_bono": False},
}

DEFAULT_DEPTH = 150
MIN_HISTORIA_PREVIA = 60
TOP_K_SIM = 20


def _safe_json(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj

def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_safe_json)


def load_history(name: str) -> pd.DataFrame:
    path = os.path.join(_DATA_DIR, "%s.csv" % name.lower())
    if not os.path.exists(path):
        raise RuntimeError("CSV no encontrado: %s" % path)
    df = pd.read_csv(path)
    df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FECHA"])
    df = df.sort_values("FECHA", ascending=True).reset_index(drop=True)
    return df


def _build_quick_stats(df_hasta_aqui: pd.DataFrame, n_max: int, k: int) -> Dict:
    ncols = ["N%d" % i for i in range(1, k + 1)]
    n = len(df_hasta_aqui)

    freq = {}
    ultimos30 = df_hasta_aqui.tail(30)
    for idx, (_, row) in enumerate(ultimos30.iloc[::-1].iterrows()):
        wt = 0.95 ** idx
        for c in ncols:
            if pd.notna(row.get(c)):
                num = int(row[c])
                freq[num] = freq.get(num, 0.) + wt
    tot = sum(freq.values()) or 1.
    hot30 = {num: v / tot for num, v in freq.items()}

    last_seen = {}
    for idx, (_, row) in enumerate(df_hasta_aqui.iterrows()):
        for c in ncols:
            if pd.notna(row.get(c)):
                v = int(row[c])
                if 1 <= v <= n_max:
                    last_seen[v] = idx
    exp_g = n_max / k
    gap = {}
    for num in range(1, n_max + 1):
        if num in last_seen:
            gap[num] = float(n - 1 - last_seen[num]) / exp_g
        else:
            gap[num] = float(n) / exp_g

    cooc = np.zeros((n_max + 1, n_max + 1), dtype=np.float32)
    for _, row in df_hasta_aqui.iterrows():
        nums = [int(row[c]) for c in ncols
                if pd.notna(row.get(c)) and 1 <= int(row[c]) <= n_max]
        if len(nums) != k:
            continue
        for a in range(len(nums)):
            for b in range(a + 1, len(nums)):
                cooc[nums[a], nums[b]] += 1
                cooc[nums[b], nums[a]] += 1
    exp_c = 2 * n * k * (k - 1) / (n_max * (n_max - 1)) or 1e-9
    cooc_norm = cooc / (exp_c + 1e-9)

    return {"hot30": hot30, "gap": gap, "cooc_norm": cooc_norm}


def _score_combo_quick(combo: Tuple[int, ...], stats: Dict) -> float:
    hot30 = stats["hot30"]
    gap   = stats["gap"]
    hot_score = sum(hot30.get(n, 0.) for n in combo)
    gap_score = sum(gap.get(n, 1.) for n in combo) * 0.1
    return float(hot_score + gap_score)


def _generar_top20_simulado(stats: Dict, n_max: int, k: int,
                             rng: np.random.Generator,
                             n_candidatos: int = 3000) -> List[Dict]:
    hot30 = stats["hot30"]
    gap   = stats["gap"]

    weights = np.ones(n_max + 1, dtype=float)
    for num, w in hot30.items():
        weights[num] += w * 8.
    for num, g in gap.items():
        weights[num] += g * 2.
    probs = weights[1:] / weights[1:].sum()
    nr = np.arange(1, n_max + 1)

    candidatos = set()
    intentos = 0
    while len(candidatos) < n_candidatos and intentos < n_candidatos * 10:
        sel = rng.choice(nr, size=k, replace=False, p=probs)
        c = tuple(sorted(int(x) for x in sel))
        if len(set(c)) == k:
            candidatos.add(c)
        intentos += 1

    scored = [(c, _score_combo_quick(c, stats)) for c in candidatos]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:TOP_K_SIM]

    if top:
        vals = [s for _, s in top]
        mn, mx = min(vals), max(vals)
        rng_v = (mx - mn) or 1.
        return [
            {"combo": list(c), "global_composite": round(1.0 + 0.2 * (s - mn) / rng_v, 5)}
            for c, s in top
        ]
    return []


def _analizar_sorteo(winning: Tuple[int, ...],
                     top20_sim: List[Dict],
                     stats: Dict) -> Dict:
    winning_set = set(winning)
    pred_combos = [tuple(sorted(p["combo"])) for p in top20_sim]
    scores      = [p["global_composite"] for p in top20_sim]

    estuvo_en_top = winning in pred_combos
    rank_en_top   = None
    if estuvo_en_top:
        rank_en_top = pred_combos.index(winning) + 1

    hot30 = stats["hot30"]
    gap   = stats["gap"]
    hot_v = [hot30.get(n, 0.) for n in winning]
    gap_v = [gap.get(n, 1.) for n in winning]

    winning_info = {
        "combo":    list(winning),
        "suma":     int(sum(winning)),
        "ml_score": round(_score_combo_quick(winning, stats) / 10., 5),
        "hot_mean": round(float(np.mean(hot_v)), 4),
        "gap_mean": round(float(np.mean(gap_v)), 4),
        "n_low":    int(sum(1 for n in winning if n <= 20)),
        "n_high":   int(sum(1 for n in winning if n >= 40)),
        "n_mid":    int(sum(1 for n in winning if 20 < n < 40)),
        "method":   "stats_only",
    }

    if estuvo_en_top:
        analysis = {
            "severity": "none",
            "miss_reasons": ["Estuvo en predicciones simuladas en posicion #%d" % rank_en_top]
        }
    else:
        min_score = min(scores) if scores else 0
        w_score_approx = _score_combo_quick(winning, stats)
        gap_rel = 1.0 if w_score_approx < min_score * 0.5 else 0.5
        severity = "grave" if gap_rel >= 1.0 else "moderado"

        razones = []
        suma = winning_info["suma"]
        if suma > 210:
            razones.append("Suma %d muy alta" % suma)
        elif suma < 115:
            razones.append("Suma %d muy baja" % suma)
        if winning_info["n_high"] >= 4:
            razones.append("4+ numeros altos (>=40)")
        if winning_info["n_low"] >= 4:
            razones.append("4+ numeros bajos (<=20)")
        if winning_info["gap_mean"] > 3.0:
            razones.append("Numeros con gap muy alto")
        if not razones:
            razones.append("Combinacion estadisticamente poco favorecida en su momento")

        analysis = {"severity": severity, "miss_reasons": razones}

    aciertos_dist = {}
    for p in top20_sim:
        m = len(set(p["combo"]) & winning_set)
        aciertos_dist[m] = aciertos_dist.get(m, 0) + 1

    return {
        "winning_info":    winning_info,
        "estuvo_en_top20": estuvo_en_top,
        "rank_en_top20":   rank_en_top,
        "analysis":        analysis,
        "aciertos_dist":   aciertos_dist,
    }


def migrar_loteria(name: str, depth: int, seed: int = 42) -> List[Dict]:
    cfg   = LOTTERIES[name]
    n_max = cfg["n_max"]
    k     = cfg["k"]

    logger.info("[%s] Cargando historico...", name)
    df = load_history(name)
    n_total = len(df)
    logger.info("[%s] %d sorteos totales disponibles", name, n_total)

    if n_total < MIN_HISTORIA_PREVIA + depth:
        depth = max(0, n_total - MIN_HISTORIA_PREVIA)
        logger.warning("[%s] Historico insuficiente, ajustando depth a %d", name, depth)

    if depth <= 0:
        logger.warning("[%s] Sin suficiente historia para migrar, se omite", name)
        return []

    rng = np.random.default_rng(seed)
    ncols = ["N%d" % i for i in range(1, k + 1)]

    inicio = n_total - depth
    entradas = []

    for idx in range(inicio, n_total):
        row = df.iloc[idx]
        try:
            winning = tuple(sorted(int(row[c]) for c in ncols
                                   if pd.notna(row.get(c))))
        except Exception:
            continue
        if len(winning) != k:
            continue

        df_previo = df.iloc[:idx]
        if len(df_previo) < MIN_HISTORIA_PREVIA:
            continue

        stats     = _build_quick_stats(df_previo, n_max, k)
        top20_sim = _generar_top20_simulado(stats, n_max, k, rng)
        if not top20_sim:
            continue

        resultado_analisis = _analizar_sorteo(winning, top20_sim, stats)

        fecha_str = row["FECHA"].strftime("%d/%m/%Y")
        bono = None
        if cfg["has_bono"] and "BONO" in df.columns and pd.notna(row.get("BONO")):
            bono = int(row["BONO"])

        entrada = {
            "lottery":         name,
            "fecha":           fecha_str,
            "winning_combo":   list(winning),
            "bono":            bono,
            "winning_info":    resultado_analisis["winning_info"],
            "estuvo_en_top20": resultado_analisis["estuvo_en_top20"],
            "rank_en_top20":   resultado_analisis["rank_en_top20"],
            "analysis":        resultado_analisis["analysis"],
            "aciertos_dist":   resultado_analisis["aciertos_dist"],
            "n_predicciones":  len(top20_sim),
            "timestamp":       datetime.now().isoformat(),
            "source":          "migration",
        }
        entradas.append(entrada)

        if (idx - inicio + 1) % 25 == 0:
            logger.info("[%s] Progreso: %d/%d sorteos migrados",
                        name, idx - inicio + 1, depth)

    logger.info("[%s] Migracion completada: %d sorteos procesados",
                name, len(entradas))
    return entradas


def main():
    parser = argparse.ArgumentParser(description="Migracion retroactiva unica")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help="Cuantos sorteos recientes migrar por loteria")
    parser.add_argument("--force", action="store_true",
                        help="Reemplazar entradas de migracion existentes")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MIGRACION RETROACTIVA - Ejecucion unica")
    logger.info("Profundidad: %d sorteos por loteria", args.depth)
    logger.info("=" * 60)

    tracking = _load_json(RETROACTIVE_FILE, {"sorteos": []})

    ya_migrado = any(e.get("source") == "migration"
                     for e in tracking.get("sorteos", []))
    if ya_migrado and not args.force:
        logger.warning(
            "Ya existe una migracion previa en retroactive_tracking.json. "
            "Usa --force para reemplazarla. Abortando para evitar duplicados."
        )
        return

    if args.force:
        tracking["sorteos"] = [
            e for e in tracking.get("sorteos", [])
            if e.get("source") != "migration"
        ]

    todas_las_entradas = []
    for name in LOTTERIES:
        entradas = migrar_loteria(name, args.depth)
        todas_las_entradas.extend(entradas)

    tracking.setdefault("sorteos", [])
    tracking["sorteos"].extend(todas_las_entradas)

    def _fecha_key(e):
        try:
            return pd.to_datetime(e.get("fecha", ""), dayfirst=True)
        except Exception:
            return pd.Timestamp.min

    tracking["sorteos"].sort(key=_fecha_key)
    tracking["migracion_ejecutada"] = datetime.now().isoformat()
    tracking["migracion_depth"]     = args.depth

    _save_json(RETROACTIVE_FILE, tracking)

    logger.info("=" * 60)
    logger.info("MIGRACION COMPLETADA")
    logger.info("Total de sorteos en tracking: %d", len(tracking["sorteos"]))
    logger.info("Archivo guardado: %s", RETROACTIVE_FILE)
    logger.info("=" * 60)

    for name in LOTTERIES:
        entradas_name = [e for e in tracking["sorteos"] if e.get("lottery") == name]
        graves = [e for e in entradas_name
                 if e.get("analysis", {}).get("severity") == "grave"]
        en_top = [e for e in entradas_name if e.get("estuvo_en_top20")]
        logger.info(
            "[%s] %d sorteos | %d en top20 simulado (%.1f%%) | %d misses graves",
            name, len(entradas_name), len(en_top),
            100 * len(en_top) / max(1, len(entradas_name)),
            len(graves)
        )


if __name__ == "__main__":
    main()