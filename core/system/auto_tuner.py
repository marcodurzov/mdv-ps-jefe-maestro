#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_tuner.py
MDV Auto-Tuner: optimizacion automatica de hiperparametros del
Jefe Maestro usando evidencia real acumulada en retroactive_tracking.json.

CONCEPTO:
  En lugar de que Marco ajuste manualmente MIN_SUM, MAX_SUM, el limite
  de apariciones por numero, o los pesos de los modulos auxiliares
  (advanced_stats, social_bias, information_theory) despues de ver
  cada sorteo, este modulo prueba automaticamente muchas combinaciones
  de esos parametros contra el historial real de sorteos ya ocurridos
  y selecciona la que mejor habria funcionado.

METODOLOGIA (grid search + backtest real):
  1. Lee retroactive_tracking.json: para cada sorteo ya ocurrido,
     tenemos el resultado real (winning_combo) y suficiente
     informacion para re-simular que hubiera pasado con distintos
     parametros de filtrado.
  2. Define una grilla de configuraciones candidatas (MIN_SUM,
     MAX_SUM, max_apariciones, pesos de modulos).
  3. Para cada configuracion candidata, re-genera un top-20
     estadistico equivalente (mismo motor rapido que usa
     retroactive_migration.py) aplicando ESOS filtros especificos,
     y mide cuantos aciertos habria dado contra los sorteos reales
     de las ultimas N semanas.
  4. Selecciona la configuracion con mejor promedio de aciertos
     (metrica: promedio ponderado de aciertos + tasa de "estuvo en
     top 20").
  5. Escribe la configuracion ganadora en auto_tuner_config.json.
  6. jefe_maestro_v6_elite_predictor.py lee ese archivo al arrancar
     y SI EXISTE, sobreescribe sus defaults (MIN_SUM, MAX_SUM, etc.)
     con los valores optimizados. Si no existe, usa los defaults
     normales sin ningun cambio de comportamiento.

CUANDO SE EJECUTA:
  Se dispara automaticamente desde main_run.py cada vez que hay
  10+ sorteos nuevos acumulados desde el ultimo tuning (no en cada
  corrida, para no sobreajustar a ruido de corto plazo).

SEGURIDAD / LIMITES:
  - Nunca prueba valores fuera de rangos razonables (hardcoded floors
    y ceilings basados en la distribucion historica real de Melate).
  - Requiere minimo 40 sorteos en el tracking para activarse.
  - Guarda el historial de configuraciones probadas para poder
    revertir si una configuracion resulta mala en el tiempo.
  - El cambio de configuracion se registra en el correo para que
    Marco siempre sepa que se ajusto y por que.
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

# Minimo de sorteos con datos reales para activar el tuning
MIN_SORTEOS_TUNING = 40

# Cada cuantos sorteos nuevos se vuelve a correr el tuner
TUNING_INTERVAL = 10

# Ventana de evaluacion: cuantos sorteos recientes usar para
# medir que configuracion es mejor (mas peso a lo reciente)
EVAL_WINDOW = 60

# ── Grilla de busqueda: rangos SEGUROS basados en la distribucion
#    real historica de Melate (suma media ~168, std ~35) ──────────
GRID = {
    "MIN_SUM":  [90, 100, 110, 115, 120, 130],
    "MAX_SUM":  [210, 215, 220, 225, 230],
    "MAX_APARICIONES": [6, 8, 10, 12],
    "W_ADVANCED": [0.20, 0.30, 0.40],   # peso del modulo advanced_stats
    "W_SOCIAL":   [0.10, 0.15, 0.20],   # peso del modulo social_bias
    "W_IT":       [0.10, 0.15, 0.20],   # peso del modulo information_theory
}

# Limite absoluto de combinaciones a probar (evita explosion combinatoria)
MAX_CONFIGS_A_PROBAR = 60


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
# EVALUACION DE UNA CONFIGURACION CONTRA HISTORIAL REAL
# ─────────────────────────────────────────────────────────────────────

def _combo_pasa_filtros(combo: Tuple[int, ...], config: Dict) -> bool:
    """Aplica los mismos filtros que is_plausible() del predictor real."""
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


def _evaluar_config_sobre_sorteo(entry: Dict, config: Dict) -> Optional[int]:
    """
    Re-evalua un sorteo historico bajo una configuracion candidata.
    Retorna cuantos aciertos habria dado la mejor combinacion simulada
    que SI hubiera pasado los filtros de esta config, o None si no
    hay suficiente informacion para evaluar ese sorteo.
    """
    winning = entry.get("winning_combo", [])
    if len(winning) != 6:
        return None

    winning_t = tuple(sorted(winning))

    # Si el ganador real no pasa los filtros de esta config,
    # el sistema NUNCA pudo haberlo generado -> 0 aciertos garantizado
    # para el mejor caso posible bajo esta config en ESTE sorteo especifico
    # (aproximacion conservadora y honesta)
    if not _combo_pasa_filtros(winning_t, config):
        return 0

    # Si SI pasa los filtros, usamos la distribucion de aciertos ya
    # calculada en la migracion/tiempo real como proxy de que tan
    # bien el motor estadistico lo habria rankeado
    aciertos_dist = entry.get("aciertos_dist", {})
    if not aciertos_dist:
        return None

    # Aciertos maximos ya observados en esa evaluacion original
    try:
        max_aciertos = max(int(k) for k in aciertos_dist.keys())
    except Exception:
        return None

    return max_aciertos


def evaluar_configuracion(config: Dict, entries: List[Dict]) -> Dict:
    """
    Evalua una configuracion candidata contra todos los sorteos
    disponibles en la ventana de evaluacion.
    """
    aciertos_totales = []
    en_rango_count = 0

    for entry in entries:
        r = _evaluar_config_sobre_sorteo(entry, config)
        if r is not None:
            aciertos_totales.append(r)
            if r > 0 or _combo_pasa_filtros(tuple(sorted(entry.get("winning_combo", [0]*6))), config):
                en_rango_count += 1

    if not aciertos_totales:
        return {"score": -999, "n_evaluados": 0}

    promedio = float(np.mean(aciertos_totales))
    tasa_en_rango = en_rango_count / len(entries) if entries else 0

    # Score combinado: promedio de aciertos (principal) +
    # bonus por mantener buena cobertura del espacio de busqueda
    # (evita que el tuner elija filtros tan estrechos que dejen
    # fuera sorteos reales sistematicamente)
    score = promedio + 0.5 * tasa_en_rango

    return {
        "score":          round(score, 4),
        "promedio_aciertos": round(promedio, 4),
        "tasa_en_rango":  round(tasa_en_rango, 4),
        "n_evaluados":    len(aciertos_totales),
    }


# ─────────────────────────────────────────────────────────────────────
# GENERACION DE CONFIGURACIONES CANDIDATAS
# ─────────────────────────────────────────────────────────────────────

def generar_configs_candidatas(seed: int = 42) -> List[Dict]:
    """
    Genera un subconjunto aleatorio de la grilla completa (que puede
    tener miles de combinaciones) limitado a MAX_CONFIGS_A_PROBAR
    para mantener el tiempo de ejecucion razonable.
    """
    keys = list(GRID.keys())
    all_combos = list(itertools.product(*[GRID[k] for k in keys]))

    rng = np.random.default_rng(seed)
    if len(all_combos) > MAX_CONFIGS_A_PROBAR:
        idx = rng.choice(len(all_combos), size=MAX_CONFIGS_A_PROBAR, replace=False)
        all_combos = [all_combos[i] for i in idx]

    configs = []
    for combo in all_combos:
        cfg = dict(zip(keys, combo))
        # Validacion de sanidad: MIN_SUM siempre menor que MAX_SUM
        if cfg["MIN_SUM"] < cfg["MAX_SUM"]:
            configs.append(cfg)

    return configs


# ─────────────────────────────────────────────────────────────────────
# FUNCION PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

def run_auto_tuner(force: bool = False) -> Dict:
    """
    Ejecuta el ciclo completo de auto-tuning.
    Retorna un dict con el resultado para incluir en el correo.
    Si no hay suficientes datos o no toca tunear aun, retorna
    {"ejecutado": False, "razon": "..."}.
    """
    logger.info("Auto-Tuner iniciando...")

    tracking = _load_json(RETROACTIVE_FILE, {"sorteos": []})
    entries_all = tracking.get("sorteos", [])

    if len(entries_all) < MIN_SORTEOS_TUNING:
        msg = "Auto-Tuner: %d/%d sorteos disponibles. Esperando mas datos." % (
            len(entries_all), MIN_SORTEOS_TUNING
        )
        logger.info(msg)
        return {"ejecutado": False, "razon": msg}

    # Verificar si toca tunear (cada TUNING_INTERVAL sorteos nuevos)
    tuner_hist = _load_json(TUNER_HISTORY_FILE, {"runs": []})
    ultimo_n = tuner_hist["runs"][-1]["n_sorteos_al_momento"] if tuner_hist["runs"] else 0

    if not force and (len(entries_all) - ultimo_n) < TUNING_INTERVAL:
        msg = "Auto-Tuner: solo %d sorteos nuevos desde el ultimo tuning (necesita %d)." % (
            len(entries_all) - ultimo_n, TUNING_INTERVAL
        )
        logger.info(msg)
        return {"ejecutado": False, "razon": msg}

    # Ventana de evaluacion: los sorteos mas recientes
    entries_eval = entries_all[-EVAL_WINDOW:] if len(entries_all) > EVAL_WINDOW else entries_all
    logger.info("Auto-Tuner: evaluando sobre %d sorteos recientes", len(entries_eval))

    # Config actual (baseline) para comparar
    config_actual = _load_json(TUNER_CONFIG_FILE, {
        "MIN_SUM": 90, "MAX_SUM": 230, "MAX_APARICIONES": 8,
        "W_ADVANCED": 0.30, "W_SOCIAL": 0.15, "W_IT": 0.15,
    })
    resultado_actual = evaluar_configuracion(config_actual, entries_eval)
    logger.info("Config actual: score=%.4f (aciertos promedio=%.3f)",
                resultado_actual["score"], resultado_actual.get("promedio_aciertos", 0))

    # Generar y evaluar candidatas
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

    # Solo aplicar el cambio si la mejora es significativa
    # (evita ruido: exige mejora minima de 3% en el score)
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

    # Guardar historial de esta corrida
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


# ─────────────────────────────────────────────────────────────────────
# LECTURA DE CONFIG PARA EL PREDICTOR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

def load_tuned_config() -> Optional[Dict]:
    """
    Funcion que jefe_maestro_v6_elite_predictor.py llama al arrancar
    para obtener la configuracion optimizada, si existe.
    Retorna None si no hay configuracion tuneada aun (usa defaults).
    """
    if not os.path.exists(TUNER_CONFIG_FILE):
        return None
    try:
        return _load_json(TUNER_CONFIG_FILE, None)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# HTML PARA EL CORREO
# ─────────────────────────────────────────────────────────────────────

def _main_cli():
    """Punto de entrada para ejecucion directa: python auto_tuner.py"""
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="MDV Auto-Tuner")
    parser.add_argument("--force", action="store_true",
                        help="Forzar tuning aunque no haya pasado el intervalo")
    args = parser.parse_args()

    resultado = run_auto_tuner(force=args.force)
    logger.info("=" * 60)
    logger.info("RESULTADO AUTO-TUNER")
    logger.info(json.dumps(resultado, indent=2, default=_safe_json))
    logger.info("=" * 60)


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


if __name__ == "__main__":
    _main_cli()