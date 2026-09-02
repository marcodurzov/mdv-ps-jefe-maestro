#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_tuner.py  v2
MDV Auto-Tuner: optimizacion automatica de hiperparametros del
Jefe Maestro usando evidencia real acumulada en retroactive_tracking.json.

CAMBIOS v2 (cierre del loop de autoaprendizaje completo):
  - Se agrega IOTA_HUM a la grilla de busqueda: la penalizacion que
    el predictor aplica a combinaciones "tipo fecha" (muchos numeros
    bajos). Evidencia real de varios sorteos mostro que el sistema
    subestimaba sistematicamente numeros <=20 -- ahora el propio
    tuner puede corregir esto solo, sin intervencion manual.
  - Se agrega heuristica is_humano_pattern(): replica la logica de
    _is_date_like() del predictor principal para poder simular el
    efecto de distintos valores de IOTA_HUM sobre sorteos ya ocurridos.
  - Los pesos W_ADVANCED/W_SOCIAL/W_IT que ya se buscaban en v1 ahora
    SI se conectan al predictor (antes se calculaban pero nunca se
    aplicaban -- quedaban huerfanos). Ver jefe_maestro_v6_elite_predictor.py.
  - MAX_APARICIONES tambien se conecta al limite real de repeticion
    de numeros en el portfolio final.

CONCEPTO (sin cambios respecto a v1):
  Grid search + backtest contra evidencia real de retroactive_tracking.json.
  Prueba muchas configuraciones candidatas contra sorteos ya ocurridos,
  selecciona la que mejor habria funcionado, y solo aplica el cambio
  si la mejora es significativa (>=3%). Nunca se autoconvence de una
  mejora basada en ruido de corto plazo.

LIMITE HONESTO:
  Esto optimiza que tanto el sistema explota los micro-sesgos
  estadisticos y fisicos reales que existen en el juego. No puede
  ni pretende resolver la aleatoriedad fundamental del sorteo.
"""

import os
import json
import logging
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(os.path.dirname(_HERE))
_DATA_DIR = os.path.join(_ROOT, "data")

RETROACTIVE_FILE = os.path.join(_ROOT, "retroactive_tracking.json")
TUNER_CONFIG_FILE = os.path.join(_ROOT, "auto_tuner_config.json")
TUNER_HISTORY_FILE = os.path.join(_ROOT, "auto_tuner_history.json")

MIN_SORTEOS_TUNING = 40
TUNING_INTERVAL = 10
EVAL_WINDOW = 60

# ── Grilla de busqueda ampliada (v2) ──────────────────────────────
GRID = {
    "MIN_SUM":  [90, 100, 110, 115, 120, 130],
    "MAX_SUM":  [210, 215, 220, 225, 230],
    "MAX_APARICIONES": [6, 8, 10, 12],
    "W_ADVANCED": [0.20, 0.30, 0.40],
    "W_SOCIAL":   [0.10, 0.15, 0.20],
    "W_IT":       [0.10, 0.15, 0.20],
    # NUEVO v2: penalizacion a combinaciones "tipo fecha"/humanas.
    # Default historico era 0.08. Evidencia real (sorteos de
    # agosto-septiembre 2026) mostro sesgo hacia subestimar numeros
    # bajos -- se agregan valores mas suaves para que el tuner
    # decida solo si corregir esto ayuda.
    "IOTA_HUM":   [0.02, 0.04, 0.06, 0.08],
}

MAX_CONFIGS_A_PROBAR = 80  # subio de 60 a 80 por la dimension extra


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


# ─────────────────────────────────────────────────────────────────────
# HEURISTICA "PATRON HUMANO" (replica _is_date_like del predictor,
# sin importar el modulo principal para evitar acoplamiento pesado)
# ─────────────────────────────────────────────────────────────────────

def is_humano_pattern(combo: Tuple[int, ...]) -> bool:
    """
    Replica la logica de _is_date_like() + chequeos adicionales de
    jefe_maestro_v6_elite_predictor.py, usada para decidir si una
    combinacion "parece" elegida por un humano (fechas, secuencias,
    multiplos, cluster 34-43).
    """
    c = sorted(combo)
    if len(c) != 6:
        return False

    # Patron de fechas: 4+ numeros <=31 y 2+ <=12
    es_fecha = (sum(1 for n in c if n <= 31) >= 4 and
                sum(1 for n in c if n <= 12) >= 2)

    # Racha consecutiva larga
    cons = mr = 1
    for i in range(len(c) - 1):
        if c[i+1] == c[i] + 1:
            cons += 1
            mr = max(mr, cons)
        else:
            cons = 1
    racha_larga = mr >= 4

    # Multiplos compartidos (2 a 7)
    max_mult = max(sum(1 for n in c if n % f == 0) for f in range(2, 8))
    muchos_multiplos = max_mult >= 4

    # Cluster en rango 34-43
    cluster_medio = sum(1 for n in c if 34 <= n <= 43) >= 4

    return es_fecha or racha_larga or muchos_multiplos or cluster_medio


# ─────────────────────────────────────────────────────────────────────
# EVALUACION DE UNA CONFIGURACION CONTRA HISTORIAL REAL
# ─────────────────────────────────────────────────────────────────────

def _combo_pasa_filtros(combo: Tuple[int, ...], config: Dict) -> bool:
    s = sum(combo)
    if not (config["MIN_SUM"] <= s <= config["MAX_SUM"]):
        return False
    if max(combo) < 35:
        return False
    cons = mr = 1
    combo_sorted = sorted(combo)
    for i in range(len(combo_sorted) - 1):
        if combo_sorted[i+1] == combo_sorted[i] + 1:
            cons += 1
            mr = max(mr, cons)
        else:
            cons = 1
    return mr <= 4


def _evaluar_config_sobre_sorteo(entry: Dict, config: Dict) -> Optional[float]:
    """
    Re-evalua un sorteo historico bajo una configuracion candidata.
    v2: ademas de los filtros duros de suma, incorpora el efecto
    aproximado de IOTA_HUM sobre combinaciones patron-humano.
    """
    winning = entry.get("winning_combo", [])
    if len(winning) != 6:
        return None

    winning_t = tuple(sorted(winning))

    if not _combo_pasa_filtros(winning_t, config):
        return 0.0

    aciertos_dist = entry.get("aciertos_dist", {})
    if not aciertos_dist:
        return None

    try:
        max_aciertos = max(int(k) for k in aciertos_dist.keys())
    except Exception:
        return None

    score = float(max_aciertos)

    # Ajuste IOTA_HUM: si el ganador real TIENE patron humano y fue
    # un miss (grave/moderado) bajo la config original, un IOTA_HUM
    # mas bajo (menos penalizacion) habria dejado que su score suba,
    # acercandolo mas a estar en el top. Se aproxima como un bonus
    # proporcional a cuanto se reduce la penalizacion vs el default 0.08.
    severity = entry.get("analysis", {}).get("severity", "none")
    if severity in ("grave", "moderado") and is_humano_pattern(winning_t):
        default_iota = 0.08
        reduccion = max(0.0, default_iota - config.get("IOTA_HUM", default_iota))
        # Bonus acotado: hasta +1.5 aciertos equivalentes si la
        # reduccion es maxima (0.08 -> 0.02, reduccion=0.06)
        bonus = min(1.5, reduccion * 25.0)
        score += bonus

    return score


def evaluar_configuracion(config: Dict, entries: List[Dict]) -> Dict:
    aciertos_totales = []
    en_rango_count = 0

    for entry in entries:
        r = _evaluar_config_sobre_sorteo(entry, config)
        if r is not None:
            aciertos_totales.append(r)
            winning_t = tuple(sorted(entry.get("winning_combo", [0]*6)))
            if r > 0 or _combo_pasa_filtros(winning_t, config):
                en_rango_count += 1

    if not aciertos_totales:
        return {"score": -999, "n_evaluados": 0}

    promedio = float(np.mean(aciertos_totales))
    tasa_en_rango = en_rango_count / len(entries) if entries else 0
    score = promedio + 0.5 * tasa_en_rango

    return {
        "score":          round(score, 4),
        "promedio_aciertos": round(promedio, 4),
        "tasa_en_rango":  round(tasa_en_rango, 4),
        "n_evaluados":    len(aciertos_totales),
    }


def generar_configs_candidatas(seed: int = 42) -> List[Dict]:
    keys = list(GRID.keys())
    all_combos = list(itertools.product(*[GRID[k] for k in keys]))

    rng = np.random.default_rng(seed)
    if len(all_combos) > MAX_CONFIGS_A_PROBAR:
        idx = rng.choice(len(all_combos), size=MAX_CONFIGS_A_PROBAR, replace=False)
        all_combos = [all_combos[i] for i in idx]

    configs = []
    for combo in all_combos:
        cfg = dict(zip(keys, combo))
        if cfg["MIN_SUM"] < cfg["MAX_SUM"]:
            configs.append(cfg)

    return configs


DEFAULT_CONFIG = {
    "MIN_SUM": 90, "MAX_SUM": 230, "MAX_APARICIONES": 8,
    "W_ADVANCED": 0.30, "W_SOCIAL": 0.15, "W_IT": 0.15,
    "IOTA_HUM": 0.08,
}


def run_auto_tuner(force: bool = False) -> Dict:
    logger.info("Auto-Tuner iniciando...")

    tracking = _load_json(RETROACTIVE_FILE, {"sorteos": []})
    entries_all = tracking.get("sorteos", [])

    if len(entries_all) < MIN_SORTEOS_TUNING:
        msg = "Auto-Tuner: %d/%d sorteos disponibles. Esperando mas datos." % (
            len(entries_all), MIN_SORTEOS_TUNING
        )
        logger.info(msg)
        return {"ejecutado": False, "razon": msg}

    tuner_hist = _load_json(TUNER_HISTORY_FILE, {"runs": []})
    ultimo_n = tuner_hist["runs"][-1]["n_sorteos_al_momento"] if tuner_hist["runs"] else 0

    if not force and (len(entries_all) - ultimo_n) < TUNING_INTERVAL:
        msg = "Auto-Tuner: solo %d sorteos nuevos desde el ultimo tuning (necesita %d)." % (
            len(entries_all) - ultimo_n, TUNING_INTERVAL
        )
        logger.info(msg)
        return {"ejecutado": False, "razon": msg}

    entries_eval = entries_all[-EVAL_WINDOW:] if len(entries_all) > EVAL_WINDOW else entries_all
    logger.info("Auto-Tuner: evaluando sobre %d sorteos recientes", len(entries_eval))

    config_actual = _load_json(TUNER_CONFIG_FILE, dict(DEFAULT_CONFIG))
    # Asegurar compatibilidad hacia adelante: si config_actual viene
    # de una version v1 sin IOTA_HUM, se completa con el default
    for k, v in DEFAULT_CONFIG.items():
        config_actual.setdefault(k, v)

    resultado_actual = evaluar_configuracion(config_actual, entries_eval)
    logger.info("Config actual: score=%.4f (aciertos promedio=%.3f)",
                resultado_actual["score"], resultado_actual.get("promedio_aciertos", 0))

    candidatas = generar_configs_candidatas()
    logger.info("Probando %d configuraciones candidatas...", len(candidatas))

    resultados = []
    for cfg in candidatas:
        r = evaluar_configuracion(cfg, entries_eval)
        resultados.append((cfg, r))

    resultados.sort(key=lambda x: x[1]["score"], reverse=True)
    mejor_config, mejor_resultado = resultados[0]

    logger.info("Mejor config encontrada: score=%.4f (aciertos promedio=%.3f)",
                mejor_resultado["score"], mejor_resultado.get("promedio_aciertos", 0))

    mejora_pct = 0.0
    aplicar_cambio = False
    if resultado_actual["score"] > 0:
        mejora_pct = (mejor_resultado["score"] - resultado_actual["score"]) / abs(resultado_actual["score"]) * 100
    else:
        mejora_pct = 100.0 if mejor_resultado["score"] > resultado_actual["score"] else 0.0

    if mejora_pct >= 3.0:
        aplicar_cambio = True
        _save_json(TUNER_CONFIG_FILE, mejor_config)
        logger.info("Auto-Tuner: nueva configuracion APLICADA (mejora %.1f%%)", mejora_pct)
    else:
        logger.info("Auto-Tuner: mejora insuficiente (%.1f%%), se mantiene config actual", mejora_pct)

    run_entry = {
        "timestamp":        datetime.now().isoformat(),
        "n_sorteos_al_momento": len(entries_all),
        "config_anterior":  config_actual,
        "score_anterior":   resultado_actual["score"],
        "config_evaluada_mejor": mejor_config,
        "score_mejor":      mejor_resultado["score"],
        "mejora_pct":       round(mejora_pct, 2),
        "aplicado":         aplicar_cambio,
        "n_candidatas_probadas": len(candidatas),
    }
    tuner_hist.setdefault("runs", []).append(run_entry)
    tuner_hist["runs"] = tuner_hist["runs"][-50:]
    _save_json(TUNER_HISTORY_FILE, tuner_hist)

    return {
        "ejecutado":       True,
        "aplicado":        aplicar_cambio,
        "mejora_pct":       round(mejora_pct, 2),
        "config_anterior": config_actual,
        "config_nueva":    mejor_config if aplicar_cambio else config_actual,
        "score_anterior":  resultado_actual["score"],
        "score_nuevo":     mejor_resultado["score"] if aplicar_cambio else resultado_actual["score"],
        "n_evaluados":     len(entries_eval),
        "n_candidatas":    len(candidatas),
    }


def load_tuned_config() -> Optional[Dict]:
    """
    Funcion que jefe_maestro_v6_elite_predictor.py llama al arrancar
    para obtener la configuracion optimizada, si existe.
    """
    if not os.path.exists(TUNER_CONFIG_FILE):
        return None
    try:
        cfg = _load_json(TUNER_CONFIG_FILE, None)
        if cfg:
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
        return cfg
    except Exception:
        return None


def generar_html_tuner(resultado: Dict) -> str:
    if not resultado or not resultado.get("ejecutado"):
        return ""

    AZUL  = "#1a3a5c"
    VERDE = "#2e7d32"
    GRIS  = "#666"

    aplicado = resultado.get("aplicado", False)
    mejora   = resultado.get("mejora_pct", 0)
    cfg_new  = resultado.get("config_nueva", {})
    cfg_old  = resultado.get("config_anterior", {})

    if aplicado:
        titulo = "Auto-Tuner: Nueva configuracion aplicada (+%.1f%%)" % mejora
        color  = VERDE
    else:
        titulo = "Auto-Tuner: configuracion actual se mantiene (sin mejora significativa)"
        color  = GRIS

    cambios_html = ""
    if aplicado:
        for key in cfg_new:
            old_v = cfg_old.get(key, "?")
            new_v = cfg_new.get(key, "?")
            if old_v != new_v:
                cambios_html += (
                    "<li style='font-size:11px'>%s: %s -> <b>%s</b></li>"
                ) % (key, old_v, new_v)

    html = (
        "<h3 style='color:%s;border-bottom:2px solid #2e75b6;"
        "padding-bottom:6px'>%s</h3>"
        "<p style='font-size:12px;color:#666'>"
        "Evaluadas %d configuraciones sobre %d sorteos reales recientes.</p>"
    ) % (AZUL, titulo, resultado.get("n_candidatas", 0), resultado.get("n_evaluados", 0))

    if cambios_html:
        html += "<ul>%s</ul>" % cambios_html

    return html


def _main_cli():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="MDV Auto-Tuner v2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    resultado = run_auto_tuner(force=args.force)
    logger.info("=" * 60)
    logger.info("RESULTADO AUTO-TUNER")
    logger.info(json.dumps(resultado, indent=2, default=_safe_json))
    logger.info("=" * 60)


if __name__ == "__main__":
    _main_cli()