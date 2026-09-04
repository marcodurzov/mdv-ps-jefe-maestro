#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retroactive_learner.py v2
Aprendizaje retroactivo real para Jefe Maestro v8.

ARQUITECTURA CORRECTA (sin bugs de importacion circular):
  - NO importa de jefe_maestro_v6_elite_predictor
  - Se llama desde main_run.py DESPUES de que los modelos ya existen en cache
  - Recibe stats y modelos como parametros, no los importa
  - Compara predicciones del run ANTERIOR vs resultado real

FLUJO CORRECTO:
  1. csv_updater descarga resultados nuevos (sorteo que ocurrio ayer/anteayer)
  2. main_run carga predictions_history_v8.json (predicciones del run anterior)
  3. retroactive_learner compara esas predicciones vs resultado real
  4. Analiza donde quedo la combo ganadora
  5. Guarda aprendizaje para el proximo reentrenamiento
  6. main_run genera NUEVAS predicciones para el proximo sorteo
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

_HERE  = os.path.dirname(os.path.abspath(__file__))
_ROOT  = os.path.dirname(os.path.dirname(_HERE))
_CACHE = os.path.join(_ROOT, "cache")

RETROACTIVE_FILE  = os.path.join(_ROOT, "retroactive_tracking.json")
PREDICTIONS_HIST  = os.path.join(_ROOT, "predictions_history_v8.json")
MODEL_FILE_TPL    = os.path.join(_CACHE, "model_v8_{name}.joblib")

MISS_THRESHOLD_PCT = 0.30


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
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_safe_json)
    except Exception as e:
        logger.error("Error guardando %s: %s", path, e)


# ─────────────────────────────────────────────────────────────────────
# OBTENER RESULTADO REAL MAS RECIENTE
# ─────────────────────────────────────────────────────────────────────

def _get_latest_results(all_histories: Dict) -> Dict[str, Dict]:
    """
    Obtiene el resultado mas reciente de cada juego directamente
    de los DataFrames ya cargados en memoria.
    """
    results = {}
    for name, df in all_histories.items():
        try:
            if df is None or df.empty:
                continue
            row   = df.iloc[0]  # mas reciente (ordenado desc)
            ncols = ["N%d" % i for i in range(1, 7)]
            nums  = [int(row[c]) for c in ncols if pd.notna(row.get(c))]
            if len(nums) == 6:
                bono = None
                if "BONO" in df.columns and pd.notna(row.get("BONO")):
                    bono = int(row["BONO"])
                results[name] = {
                    "fecha":   str(row.get("FECHA", "")),
                    "numeros": sorted(nums),
                    "bono":    bono,
                }
        except Exception as e:
            logger.warning("[RL:%s] Error obteniendo resultado: %s", name, e)
    return results


# ─────────────────────────────────────────────────────────────────────
# OBTENER PREDICCIONES DEL RUN ANTERIOR
# ─────────────────────────────────────────────────────────────────────

def _get_previous_predictions(fecha_resultado: str) -> Optional[List[Dict]]:
    """
    Lee predictions_history_v8.json y obtiene las predicciones
    hechas ANTES de la fecha del resultado real.
    Esta es la lista de las 20 combinaciones que predijimos.
    """
    history = _load_json(PREDICTIONS_HIST, {})
    entries = history.get("GlobalUnified_v8", [])
    if not entries:
        return None

    fecha_res = pd.to_datetime(fecha_resultado, dayfirst=True, errors="coerce")
    if pd.isna(fecha_res):
        return None

    candidates = []
    for entry in entries:
        fp = pd.to_datetime(entry.get("date", ""), errors="coerce")
        if not pd.isna(fp) and fp < fecha_res:
            candidates.append((fp, entry))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1].get("predicted", [])


# ─────────────────────────────────────────────────────────────────────
# SCORING SIN IMPORTACION CIRCULAR
# ─────────────────────────────────────────────────────────────────────

def _score_combo_with_model(combo: tuple, name: str,
                             stats: Dict, n_max: int = 56) -> Dict:
    """
    Calcula un score heuristico (NO es el score del ensemble ML real)
    de una combinacion, basado en hot30/gap ya calculados por el
    predictor principal.

    FIX: la version anterior intentaba aproximar el score del
    ensemble ML real construyendo un vector de ~18 features y
    rellenando con ceros hasta las ~70 dimensiones que el scaler
    entrenado espera. Esto produce un vector completamente fuera
    de la distribucion de entrenamiento, y los modelos (arboles de
    decision calibrados) extrapolan de forma degenerada -- en
    produccion esto se observo como un score identico (0.0000)
    para tres loterias con modelos independientes en la misma
    corrida, lo cual es matematicamente casi imposible si el calculo
    fuera real. Confirmado tambien con simulacion controlada: el
    resultado del zero-padding no guarda relacion estable con las
    features reales de la combinacion.

    En vez de fingir precision de ML que no existe, este score usa
    directamente hot30 (frecuencia reciente) y gap (sorteos desde
    la ultima aparicion) de cada numero -- la misma señal base que
    ya usa el pipeline real, sin la capa de aproximacion rota.
    No importa el modelo desde cache: es mas rapido y mas honesto.
    """
    nums = sorted(combo)
    h30  = stats.get("hot30", {})
    hot  = [float(h30.get(str(n), 0.)) for n in nums]
    gap  = stats.get("gap", {})
    gaps = [float(gap.get(str(n), 1.)) for n in nums]

    # Score heuristico: combina frecuencia reciente alta con gap bajo
    # (numeros calientes y recientes). Rango aproximado 0-1.
    hot_mean = float(np.mean(hot))
    gap_mean = float(np.mean(gaps))
    score = float(np.clip(hot_mean * 5.0 - gap_mean * 0.05, 0.0, 1.0))

    return {
        "combo":    list(combo),
        "suma":     int(sum(combo)),
        "ml_score": round(score, 5),
        "hot_mean": round(hot_mean, 4),
        "gap_mean": round(gap_mean, 4),
        "n_low":    int(sum(1 for n in nums if n <= 20)),
        "n_high":   int(sum(1 for n in nums if n >= 40)),
        "n_mid":    int(sum(1 for n in nums if 20 < n < 40)),
        "method":   "heuristico",
    }


# ─────────────────────────────────────────────────────────────────────
# ANALISIS DEL MISS
# ─────────────────────────────────────────────────────────────────────

def _analyze_miss(winning_info: Dict,
                  top20_scores: List[float],
                  previous_preds: Optional[List[Dict]]) -> Dict:
    analysis = {"miss_reasons": [], "severity": "none"}

    if not top20_scores:
        return analysis

    min_top20 = min(top20_scores)
    w_score   = winning_info.get("ml_score", 0)
    gap       = min_top20 - w_score

    if gap <= 0:
        analysis["severity"]     = "none"
        analysis["miss_reasons"] = ["La combo ganadora SI estuvo en las predicciones"]
        return analysis

    analysis["gap_to_predictions"] = round(gap, 4)
    analysis["severity"] = "grave" if gap > 0.08 else ("moderado" if gap > 0.03 else "leve")

    suma = winning_info.get("suma", 0)
    if suma > 210:
        analysis["miss_reasons"].append("Suma %d muy alta — el modelo la filtra o penaliza" % suma)
    elif suma < 115:
        analysis["miss_reasons"].append("Suma %d muy baja — el modelo la penaliza" % suma)

    if winning_info.get("n_high", 0) >= 4:
        analysis["miss_reasons"].append("4+ numeros altos (>=40): subrepresentados en entrenamiento")
    if winning_info.get("n_low", 0) >= 4:
        analysis["miss_reasons"].append("4+ numeros bajos (<=20): penalizados por sesgo de fechas")

    gap_mean = winning_info.get("gap_mean", 1)
    if gap_mean > 3.0:
        analysis["miss_reasons"].append(
            "Numeros con gap muy alto (%.1f): llevan mucho tiempo sin salir" % gap_mean
        )

    if not analysis["miss_reasons"]:
        analysis["miss_reasons"].append(
            "Score marginalmente bajo (gap=%.4f) — varianza normal del modelo" % gap
        )

    return analysis


def _detect_patterns(history: List[Dict]) -> Dict:
    if len(history) < 5:
        return {"disponible": False}

    graves = [e for e in history if e.get("analysis",{}).get("severity") == "grave"]
    if not graves:
        return {"disponible": True, "grave_misses": 0,
                "patron": "Sin misses graves — modelo bien calibrado"}

    sumas  = [e.get("winning_info",{}).get("suma", 0) for e in graves]
    nhigh  = [e.get("winning_info",{}).get("n_high", 0) for e in graves]
    nlow   = [e.get("winning_info",{}).get("n_low", 0) for e in graves]

    sesgos = []
    if np.mean(sumas) > 195:  sesgos.append("subestima sumas altas (>190)")
    if np.mean(sumas) < 140:  sesgos.append("subestima sumas bajas (<140)")
    if np.mean(nhigh) > 2.5:  sesgos.append("subestima numeros altos (>=40)")
    if np.mean(nlow) > 2.5:   sesgos.append("subestima numeros bajos (<=20)")

    return {
        "disponible":        True,
        "n_analizados":      len(history),
        "grave_misses":      len(graves),
        "avg_suma_miss":     round(float(np.mean(sumas)), 1),
        "sesgos_detectados": sesgos or ["Sin sesgo sistematico claro"],
    }


def _save_reinforced(name: str, history: List[Dict]):
    """Guarda combos ganadoras que el modelo fallo, como ejemplos para reentrenamiento."""
    graves = [e for e in history
              if e.get("lottery") == name
              and e.get("analysis",{}).get("severity") == "grave"]

    if len(graves) < 3:
        return

    examples = [{"combo": e["winning_combo"], "weight": 3.0, "fecha": e["fecha"]}
                for e in graves[-20:] if len(e.get("winning_combo",[])) == 6]

    if examples:
        path = os.path.join(_ROOT, "reinforced_%s.json" % name.lower())
        _save_json(path, {"name": name, "updated": datetime.now().isoformat(),
                          "n": len(examples), "examples": examples})
        logger.info("[RL:%s] %d ejemplos reforzados guardados", name, len(examples))


# ─────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

def run_retroactive_learning(all_histories: Dict,
                              all_stats: Dict) -> Dict:
    """
    Punto de entrada principal. Llamar desde main_run.py
    ANTES de generar nuevas predicciones.

    Args:
        all_histories: DataFrames cargados (Melate, Revancha, Revanchita)
        all_stats:     Stats calculadas para cada juego

    Returns:
        Dict con reporte para incluir en el correo
    """
    logger.info("Retroactive Learner v2 iniciando...")

    tracking = _load_json(RETROACTIVE_FILE, {"sorteos": []})
    reporte  = {}

    # Obtener resultados reales mas recientes
    resultados = _get_latest_results(all_histories)

    for name, resultado in resultados.items():
        fecha   = resultado["fecha"]
        winning = tuple(sorted(resultado["numeros"]))

        # Verificar si ya analizamos este sorteo
        ya = any(e.get("lottery") == name and e.get("fecha") == fecha
                 for e in tracking.get("sorteos", []))
        if ya:
            logger.info("[RL:%s] %s ya analizado", name, fecha)
            # Incluir en reporte con datos guardados
            for e in tracking.get("sorteos", []):
                if e.get("lottery") == name and e.get("fecha") == fecha:
                    reporte[name] = {
                        "winning_combo":   e.get("winning_combo", []),
                        "estuvo_en_top20": e.get("estuvo_en_top20", False),
                        "rank_en_top20":   e.get("rank_en_top20"),
                        "severidad":       e.get("analysis",{}).get("severity","?"),
                        "razones":         e.get("analysis",{}).get("miss_reasons",[]),
                        "suma_ganadora":   e.get("winning_info",{}).get("suma",0),
                        "ml_score_ganador":e.get("winning_info",{}).get("ml_score",0),
                        "patrones":        _detect_patterns(tracking["sorteos"]),
                    }
            continue

        # Predicciones del run anterior para este sorteo
        prev_preds = _get_previous_predictions(fecha)
        if not prev_preds:
            logger.info("[RL:%s] Sin predicciones previas para %s", name, fecha)
            continue

        # Verificar si la combo ganadora estaba en nuestras predicciones
        pred_combos    = [tuple(sorted(int(n) for n in p.get("combo",[])))
                          for p in prev_preds]
        top20_scores   = [float(p.get("global_composite", 0)) for p in prev_preds]
        estuvo_en_top  = winning in pred_combos
        rank_en_top    = None
        if estuvo_en_top:
            for i, c in enumerate(pred_combos):
                if c == winning:
                    rank_en_top = i + 1
                    break

        # Score de la combo ganadora
        stats       = all_stats.get(name, {})
        winning_info = _score_combo_with_model(winning, name, stats)
        winning_info["plausible"] = (
            115 <= winning_info.get("suma", 0) <= 225 and
            winning_info.get("n_high", 0) + winning_info.get("n_mid", 0) > 0
        )

        # Analisis del miss
        analysis = _analyze_miss(winning_info, top20_scores, prev_preds)
        if estuvo_en_top:
            analysis = {
                "severity":     "none",
                "miss_reasons": ["Estuvo en predicciones en posicion #%d" % rank_en_top]
            }

        # Aciertos reales (cuantos numeros del ganador estaban en cada prediccion)
        aciertos_dist = {}
        for p in prev_preds:
            pset = set(int(n) for n in p.get("combo",[]))
            m    = len(pset & set(winning))
            aciertos_dist[m] = aciertos_dist.get(m, 0) + 1

        entry = {
            "lottery":         name,
            "fecha":           fecha,
            "winning_combo":   list(winning),
            "winning_info":    winning_info,
            "estuvo_en_top20": estuvo_en_top,
            "rank_en_top20":   rank_en_top,
            "analysis":        analysis,
            "aciertos_dist":   aciertos_dist,
            "n_predicciones":  len(prev_preds),
            "top20_score_min": round(min(top20_scores), 5) if top20_scores else 0,
            "timestamp":       datetime.now().isoformat(),
        }
        tracking.setdefault("sorteos", []).append(entry)

        patrones = _detect_patterns(tracking["sorteos"])
        _save_reinforced(name, tracking["sorteos"])

        reporte[name] = {
            "winning_combo":    list(winning),
            "estuvo_en_top20":  estuvo_en_top,
            "rank_en_top20":    rank_en_top,
            "severidad":        analysis.get("severity", "?"),
            "razones":          analysis.get("miss_reasons", []),
            "suma_ganadora":    winning_info.get("suma", 0),
            "ml_score_ganador": winning_info.get("ml_score", 0),
            "aciertos_dist":    aciertos_dist,
            "patrones":         patrones,
        }

        logger.info(
            "[RL:%s] %s | Ganador: %s | En top: %s | Rank: %s | Severidad: %s",
            name, fecha, list(winning), estuvo_en_top, rank_en_top,
            analysis.get("severity","?")
        )

    # Guardar
    tracking["sorteos"]       = tracking["sorteos"][-100:]
    tracking["ultimo_update"] = datetime.now().isoformat()
    _save_json(RETROACTIVE_FILE, tracking)

    logger.info("Retroactive Learner completado: %d juegos", len(reporte))
    return reporte


# ─────────────────────────────────────────────────────────────────────
# HTML PARA EL CORREO
# ─────────────────────────────────────────────────────────────────────

def generar_html_retroactivo(reporte: Dict) -> str:
    if not reporte:
        return ""

    AZUL    = "#1a3a5c"
    VERDE   = "#2e7d32"
    ROJO    = "#c62828"
    NARANJA = "#e65100"

    def color_sev(s):
        return VERDE if s == "none" else (ROJO if s == "grave" else NARANJA)

    html = (
        "<h3 style='color:%s;border-bottom:2px solid #2e75b6;"
        "padding-bottom:6px'>Aprendizaje Retroactivo</h3>"
        "<p style='font-size:12px;color:#666'>"
        "El sistema analiza donde quedo la combinacion ganadora "
        "real entre nuestras predicciones y por que no subio al top 5.</p>" % AZUL
    )

    for name, data in reporte.items():
        combo_str  = " ".join("%02d" % n for n in data.get("winning_combo", []))
        en_top     = data.get("estuvo_en_top20", False)
        rank_t20   = data.get("rank_en_top20")
        sev        = data.get("severidad", "?")
        razones    = data.get("razones", [])
        ml_score   = data.get("ml_score_ganador", 0)
        suma       = data.get("suma_ganadora", 0)
        patrones   = data.get("patrones", {})
        aciertos   = data.get("aciertos_dist", {})

        if en_top:
            top_txt   = "Estuvo en predicciones — posicion #%s" % rank_t20
            top_color = VERDE
        else:
            top_txt   = "NO estuvo en nuestras 20 predicciones"
            top_color = ROJO

        razones_html = "".join("<li style='font-size:11px'>%s</li>" % r for r in razones)

        # Distribucion de aciertos
        dist_html = ""
        for m in sorted(aciertos.keys(), reverse=True):
            n = aciertos[m]
            if n > 0:
                bar = "█" * n
                color = VERDE if m >= 3 else (NARANJA if m >= 2 else "#999")
                dist_html += (
                    "<span style='font-size:11px'>&nbsp;&nbsp;%d aciertos: "
                    "<span style='color:%s'>%s</span> (%d combos)</span><br>"
                ) % (m, color, bar, n)

        sesgos    = patrones.get("sesgos_detectados", [])
        sesgos_txt = ""
        if patrones.get("disponible") and patrones.get("grave_misses", 0) > 0:
            sesgos_txt = (
                "<br><span style='font-size:11px;color:%s'>"
                "Sesgo detectado (%d misses graves): %s</span>"
            ) % (ROJO, patrones["grave_misses"], " | ".join(sesgos))

        html += (
            "<div style='background:#f9f9f9;border-left:4px solid %s;"
            "border-radius:4px;padding:10px;margin-bottom:10px'>"
            "<b style='color:%s'>%s</b><br>"
            "<b>Ganador real:</b> <span style='letter-spacing:2px'>"
            "<b>%s</b></span> (suma: %d | score heuristico: %.4f)<br>"
            "<span style='color:%s'><b>%s</b></span><br><br>"
            "<b style='font-size:11px'>Por que no quedo en top 5:</b>"
            "<ul>%s</ul>"
            "<b style='font-size:11px'>Aciertos en nuestras 20 predicciones:</b><br>%s"
            "%s"
            "</div>"
        ) % (
            color_sev(sev), AZUL, name,
            combo_str, suma, ml_score,
            top_color, top_txt,
            razones_html,
            dist_html,
            sesgos_txt,
        )

    html += (
        "<p style='font-size:10px;color:#aaa'>"
        "Con 3+ misses graves del mismo tipo, el sistema agrega esas "
        "combinaciones como ejemplos de entrenamiento con peso 3x "
        "para el siguiente ciclo de aprendizaje.</p>"
    )
    return html