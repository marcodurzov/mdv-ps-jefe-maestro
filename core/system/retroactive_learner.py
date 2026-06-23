#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retroactive_learner.py
Aprendizaje retroactivo real para Jefe Maestro v8.

CONCEPTO CENTRAL:
  Actualmente Rocket Phoenix compara solo las TOP 20 predicciones
  vs el resultado real. Si la combinacion ganadora quedo en el lugar
  #847 de 160,000 candidatos, esa informacion se pierde.

  Este modulo hace lo que un analista humano haria:
  1. Toma la combinacion ganadora real
  2. La pasa por el pipeline completo de scoring
  3. Encuentra en que posicion hubiera quedado
  4. Analiza por que sus features la hicieron bajar
  5. Usa eso como senal de entrenamiento fuerte

APRENDIZAJES IMPLEMENTADOS:
  A. Rank retroactivo: donde hubiera quedado la combinacion ganadora
  B. Feature analysis: que features la penalizaron
  C. Miss penalty: si el ganador quedo muy bajo, agregar como
     ejemplo positivo reforzado en el proximo reentrenamiento
  D. Pattern detection: detectar patrones en combinaciones ganadoras
     que el modelo sistematicamente subestima

INTEGRACION:
  Se llama desde rocket_phoenix.py despues de trackear aciertos.
  Los datos se guardan en retroactive_tracking.json
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

RETROACTIVE_FILE = os.path.join(_ROOT, "retroactive_tracking.json")
STATS_FILE_TPL   = os.path.join(_CACHE, "stats_v8_{name}.joblib")
MODEL_FILE_TPL   = os.path.join(_CACHE, "model_v8_{name}.joblib")

# Umbral: si la combo ganadora quedo por debajo de esta posicion
# entre los candidatos evaluados, se considera un "miss grave"
MISS_THRESHOLD_PCT = 0.30  # top 30% = miss leve, abajo = miss grave


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
# SCORING DE UNA COMBINACION ESPECIFICA
# ─────────────────────────────────────────────────────────────────────

def _score_single_combo(combo: tuple, name: str, stats: Dict,
                         n_max: int = 56) -> Dict:
    """
    Calcula el score completo de una combinacion especifica
    usando los mismos modelos que el pipeline principal.
    """
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from jefe_maestro_v6_elite_predictor import (
            combo_features, _stats_args, is_plausible
        )

        mf = MODEL_FILE_TPL.format(name=name)
        if not os.path.exists(mf):
            return {"error": "modelo no disponible"}

        cached = joblib.load(mf)
        scaler = cached.get("scaler")
        models = cached.get("models", {})

        # Obtener features
        args  = _stats_args(stats)
        feats = combo_features(combo, n_max, *args)
        X     = np.array([feats], dtype=np.float32)
        Xs    = scaler.transform(X) if scaler is not None else X

        # Score ensemble
        scores = {}
        total_score = 0.0
        total_weight = 0.0

        for mname, weight in [("xgb", 0.35), ("lgbm", 0.20), ("catboost", 0.15)]:
            m = models.get(mname)
            if m:
                try:
                    p = float(m.predict_proba(Xs)[0, 1])
                    scores[mname] = round(p, 4)
                    total_score  += weight * p
                    total_weight += weight
                except Exception:
                    pass

        # Meta-learner si disponible
        meta = models.get("meta")
        mc   = models.get("meta_cols", [])
        if meta and len(mc) >= 2:
            try:
                cp = [models[bn].predict_proba(Xs)[0, 1]
                      for bn in mc if models.get(bn)]
                if len(cp) == len(mc):
                    Z = np.array([cp])
                    meta_score = float(meta.predict_proba(Z)[0, 1])
                    scores["meta"] = round(meta_score, 4)
                    total_score    = meta_score
                    total_weight   = 1.0
            except Exception:
                pass

        final_score = total_score / (total_weight or 1.0)

        # Plausibilidad
        plausible = is_plausible(combo)

        # Estadisticas de features
        h30   = stats.get("hot30", {})
        hot_v = [float(h30.get(str(n), 0.)) for n in combo]
        gap_d = stats.get("gap", {})
        gap_v = [float(gap_d.get(str(n), 1.)) for n in combo]

        return {
            "combo":        list(combo),
            "suma":         int(sum(combo)),
            "ml_score":     round(final_score, 5),
            "scores_by_model": scores,
            "plausible":    plausible,
            "hot_mean":     round(float(np.mean(hot_v)), 4),
            "hot_max":      round(float(np.max(hot_v)), 4),
            "gap_mean":     round(float(np.mean(gap_v)), 4),
            "n_low":        int(sum(1 for n in combo if n <= 20)),
            "n_high":       int(sum(1 for n in combo if n >= 40)),
            "n_mid":        int(sum(1 for n in combo if 20 < n < 40)),
        }
    except Exception as e:
        logger.warning("Error scoring combo: %s", e)
        return {"error": str(e)}


def _estimate_rank(winning_score: float,
                   sample_scores: List[float]) -> Dict:
    """
    Estima el rank de la combinacion ganadora entre los candidatos.
    Usa los scores de una muestra representativa del pipeline.
    """
    if not sample_scores:
        return {"rank_estimated": -1, "percentile": -1}

    arr      = np.array(sample_scores)
    rank     = int(np.sum(arr >= winning_score)) + 1
    pct      = float(np.mean(arr < winning_score)) * 100

    return {
        "rank_estimated": rank,
        "percentile":     round(pct, 1),
        "n_sampled":      len(sample_scores),
        "score_p50":      round(float(np.percentile(arr, 50)), 4),
        "score_p75":      round(float(np.percentile(arr, 75)), 4),
        "score_p90":      round(float(np.percentile(arr, 90)), 4),
    }


# ─────────────────────────────────────────────────────────────────────
# ANALISIS DE POR QUE NO QUEDO EN TOP 20
# ─────────────────────────────────────────────────────────────────────

def _analyze_miss(winning_info: Dict, stats: Dict,
                  top20_scores: List[float]) -> Dict:
    """
    Analiza por que la combinacion ganadora no quedo en el top 20.
    Identifica que features la penalizaron vs las que quedaron arriba.
    """
    analysis = {
        "miss_reasons": [],
        "severity":     "none",
    }

    w_score = winning_info.get("ml_score", 0)
    if not top20_scores:
        return analysis

    min_top20 = min(top20_scores)
    gap_to_top20 = min_top20 - w_score

    if gap_to_top20 <= 0:
        analysis["severity"] = "none"
        analysis["miss_reasons"].append("La combinacion ganadora SI estuvo en top 20")
        return analysis

    # Severidad del miss
    if gap_to_top20 > 0.10:
        analysis["severity"] = "grave"
    elif gap_to_top20 > 0.05:
        analysis["severity"] = "moderado"
    else:
        analysis["severity"] = "leve"

    analysis["gap_to_top20"] = round(gap_to_top20, 4)

    # Razon 1: Suma fuera del rango optimo del modelo
    suma = winning_info.get("suma", 0)
    if suma < 130 or suma > 210:
        analysis["miss_reasons"].append(
            "Suma %d fuera del rango preferido del modelo (130-210)" % suma
        )

    # Razon 2: Numeros poco frecuentes
    hot_mean = winning_info.get("hot_mean", 0)
    if hot_mean < 0.01:
        analysis["miss_reasons"].append(
            "Numeros con muy baja frecuencia reciente (hot_mean=%.4f)" % hot_mean
        )

    # Razon 3: Gap alto (numeros que llevan mucho sin salir)
    gap_mean = winning_info.get("gap_mean", 1)
    if gap_mean > 2.5:
        analysis["miss_reasons"].append(
            "Numeros con alto gap - llevan mucho tiempo sin salir (gap=%.2f)" % gap_mean
        )

    # Razon 4: Distribucion extrema de numeros
    n_high = winning_info.get("n_high", 0)
    n_low  = winning_info.get("n_low", 0)
    if n_high >= 4:
        analysis["miss_reasons"].append(
            "4+ numeros altos (>=40): el modelo los subestima sistematicamente"
        )
    if n_low >= 4:
        analysis["miss_reasons"].append(
            "4+ numeros bajos (<=20): el modelo los penaliza por sesgo de fechas"
        )

    # Razon 5: Plausibilidad
    if not winning_info.get("plausible", True):
        analysis["miss_reasons"].append(
            "Filtro is_plausible la elimino antes de llegar al ranking"
        )

    if not analysis["miss_reasons"]:
        analysis["miss_reasons"].append(
            "Score marginalmente bajo - varianza normal del modelo"
        )

    return analysis


# ─────────────────────────────────────────────────────────────────────
# DETECCION DE PATRONES SISTEMATICOS
# ─────────────────────────────────────────────────────────────────────

def _detect_patterns(retroactive_history: List[Dict]) -> Dict:
    """
    Con multiples sorteos trackeados, detecta patrones sistematicos
    en las combinaciones ganadoras que el modelo subestima.
    """
    if len(retroactive_history) < 5:
        return {"disponible": False, "razon": "Menos de 5 sorteos analizados"}

    # Analizar sumas de combinaciones ganadoras que el modelo fallo
    missed_sumas = []
    missed_n_high = []
    missed_n_low  = []
    grave_misses  = 0

    for entry in retroactive_history:
        analysis = entry.get("analysis", {})
        info     = entry.get("winning_info", {})
        if analysis.get("severity") in ("grave", "moderado"):
            grave_misses  += 1
            missed_sumas.append(info.get("suma", 0))
            missed_n_high.append(info.get("n_high", 0))
            missed_n_low.append(info.get("n_low", 0))

    if not missed_sumas:
        return {
            "disponible":  True,
            "grave_misses": 0,
            "patron":      "Sin misses graves detectados - modelo bien calibrado",
        }

    avg_suma  = float(np.mean(missed_sumas))
    avg_nhigh = float(np.mean(missed_n_high))
    avg_nlow  = float(np.mean(missed_n_low))

    # Detectar si hay sesgo hacia sumas altas o bajas
    sesgo = []
    if avg_suma > 190:
        sesgo.append("modelo subestima sumas altas (>190)")
    elif avg_suma < 140:
        sesgo.append("modelo subestima sumas bajas (<140)")
    if avg_nhigh > 2.5:
        sesgo.append("modelo subestima numeros altos (>=40)")
    if avg_nlow > 2.5:
        sesgo.append("modelo subestima numeros bajos (<=20)")

    return {
        "disponible":   True,
        "n_analizados": len(retroactive_history),
        "grave_misses": grave_misses,
        "avg_suma_miss": round(avg_suma, 1),
        "avg_n_high_miss": round(avg_nhigh, 2),
        "avg_n_low_miss":  round(avg_nlow, 2),
        "sesgos_detectados": sesgo if sesgo else ["Sin sesgos sistematicos claros"],
    }


# ─────────────────────────────────────────────────────────────────────
# REENTRENAMIENTO REFORZADO
# ─────────────────────────────────────────────────────────────────────

def _reinforce_model(name: str, retroactive_history: List[Dict],
                      stats: Dict):
    """
    Agrega las combinaciones ganadoras que el modelo fallo
    como ejemplos positivos reforzados en el proximo reentrenamiento.

    Guarda un archivo de "ejemplos reforzados" que el predictor
    principal incorpora en su proximo build_dataset.
    """
    reinforced_file = os.path.join(_ROOT, "reinforced_examples_%s.json" % name)

    misses_graves = [
        entry for entry in retroactive_history
        if entry.get("lottery") == name
        and entry.get("analysis", {}).get("severity") == "grave"
    ]

    if len(misses_graves) < 3:
        return

    # Guardar las combinaciones ganadoras que fallamos como positivos
    examples = []
    for m in misses_graves[-20:]:  # ultimos 20 misses graves
        combo = m.get("winning_combo", [])
        if len(combo) == 6:
            examples.append({
                "combo":  combo,
                "weight": 3.0,  # peso 3x vs ejemplo normal
                "fecha":  m.get("fecha", ""),
                "reason": m.get("analysis", {}).get("miss_reasons", []),
            })

    if examples:
        _save_json(reinforced_file, {
            "name":     name,
            "updated":  datetime.now().isoformat(),
            "n_examples": len(examples),
            "examples": examples,
        })
        logger.info("[RL:%s] %d ejemplos reforzados guardados para reentrenamiento",
                    name, len(examples))


# ─────────────────────────────────────────────────────────────────────
# FUNCION PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

def run_retroactive_learning(
    winning_combos: Dict[str, List[int]],
    all_stats:      Dict[str, Dict],
    top20_by_name:  Dict[str, List[Dict]],
    sample_scores:  Dict[str, List[float]] = None,
) -> Dict:
    """
    Analiza por que las combinaciones ganadoras no quedaron en top 20
    y aprende de ello.

    Args:
        winning_combos:  {nombre: [n1,n2,n3,n4,n5,n6]} del sorteo real
        all_stats:       stats cargadas del predictor para cada juego
        top20_by_name:   las top 20 combinaciones predichas por juego
        sample_scores:   muestra de scores del pipeline para estimar rank

    Returns:
        Dict con el analisis completo para incluir en el correo
    """
    logger.info("Retroactive Learner iniciando...")

    tracking = _load_json(RETROACTIVE_FILE, {"sorteos": []})
    reporte  = {}

    for name, winning in winning_combos.items():
        if not winning or len(winning) != 6:
            continue

        combo  = tuple(sorted(int(n) for n in winning))
        stats  = all_stats.get(name, {})
        top20  = top20_by_name.get(name, [])
        sample = (sample_scores or {}).get(name, [])

        # 1. Score de la combinacion ganadora
        logger.info("[RL:%s] Analizando combo ganador: %s", name, list(combo))
        winning_info = _score_single_combo(combo, name, stats)

        # 2. Verificar si estaba en el top 20
        top20_combos = [tuple(sorted(int(n) for n in r.get("combo", [])))
                        for r in top20]
        estuvo_en_top20 = combo in top20_combos
        rank_en_top20   = None
        if estuvo_en_top20:
            for i, c in enumerate(top20_combos):
                if c == combo:
                    rank_en_top20 = i + 1
                    break

        # 3. Rank estimado entre todos los candidatos
        top20_scores = [r.get("global_composite", 0) for r in top20]
        rank_info = _estimate_rank(
            winning_info.get("ml_score", 0), sample or top20_scores
        )

        # 4. Analisis del miss
        analysis = _analyze_miss(winning_info, stats, top20_scores)
        if estuvo_en_top20:
            analysis["severity"]     = "none"
            analysis["miss_reasons"] = ["Estuvo en top 20 en posicion #%d" % rank_en_top20]

        # 5. Guardar en tracking
        entry = {
            "lottery":       name,
            "fecha":         datetime.now().strftime("%d/%m/%Y"),
            "winning_combo": list(combo),
            "winning_info":  winning_info,
            "estuvo_en_top20": estuvo_en_top20,
            "rank_en_top20": rank_en_top20,
            "rank_estimado": rank_info,
            "analysis":      analysis,
            "top20_score_min": round(min(top20_scores), 5) if top20_scores else 0,
            "top20_score_max": round(max(top20_scores), 5) if top20_scores else 0,
            "timestamp":     datetime.now().isoformat(),
        }
        tracking.setdefault("sorteos", []).append(entry)

        # 6. Detectar patrones sistematicos
        patrones = _detect_patterns(tracking["sorteos"])

        # 7. Reforzar modelo si hay misses graves
        _reinforce_model(name, tracking["sorteos"], stats)

        reporte[name] = {
            "winning_combo":    list(combo),
            "estuvo_en_top20":  estuvo_en_top20,
            "rank_en_top20":    rank_en_top20,
            "ml_score_ganador": winning_info.get("ml_score", 0),
            "severidad":        analysis.get("severity", "unknown"),
            "razones":          analysis.get("miss_reasons", []),
            "rank_estimado":    rank_info.get("rank_estimated", -1),
            "percentil":        rank_info.get("percentile", -1),
            "patrones":         patrones,
            "suma_ganadora":    winning_info.get("suma", 0),
        }

        logger.info(
            "[RL:%s] Ganador %s | Score: %.4f | En top20: %s | "
            "Rank estimado: #%s | Severidad: %s",
            name, list(combo),
            winning_info.get("ml_score", 0),
            estuvo_en_top20,
            rank_info.get("rank_estimated", "?"),
            analysis.get("severity", "?")
        )

    # Guardar tracking actualizado
    tracking["sorteos"]       = tracking["sorteos"][-100:]
    tracking["ultimo_update"] = datetime.now().isoformat()
    _save_json(RETROACTIVE_FILE, tracking)

    logger.info("Retroactive Learner completado. %d juegos analizados.", len(reporte))
    return reporte


# ─────────────────────────────────────────────────────────────────────
# HTML PARA EL CORREO
# ─────────────────────────────────────────────────────────────────────

def generar_html_retroactivo(reporte: Dict) -> str:
    if not reporte:
        return ""

    AZUL  = "#1a3a5c"
    VERDE = "#2e7d32"
    ROJO  = "#c62828"
    NARANJA = "#e65100"

    def color_sev(s):
        if s == "none":    return VERDE
        if s == "leve":    return NARANJA
        if s == "moderado": return NARANJA
        return ROJO

    html = (
        "<h3 style='color:%s;border-bottom:2px solid #2e75b6;"
        "padding-bottom:6px'>Aprendizaje Retroactivo - Analisis del Ganador</h3>"
        "<p style='font-size:12px;color:#666'>El sistema analiza "
        "por que la combinacion ganadora no quedo en el top 20 "
        "y ajusta su aprendizaje.</p>" % AZUL
    )

    for name, data in reporte.items():
        combo_str = " ".join("%02d" % n for n in data.get("winning_combo", []))
        en_top20  = data.get("estuvo_en_top20", False)
        rank_t20  = data.get("rank_en_top20")
        rank_est  = data.get("rank_estimado", -1)
        pct       = data.get("percentil", -1)
        sev       = data.get("severidad", "unknown")
        razones   = data.get("razones", [])
        ml_score  = data.get("ml_score_ganador", 0)
        suma      = data.get("suma_ganadora", 0)
        patrones  = data.get("patrones", {})

        if en_top20:
            top20_txt = "ESTUVO en top 20 — posicion #%s" % rank_t20
            top20_color = VERDE
        else:
            top20_txt = "NO estuvo en top 20"
            top20_color = ROJO

        rank_txt = "#%d (percentil %.0f%%)" % (rank_est, pct) \
                   if rank_est > 0 else "No calculado"

        razones_html = "".join(
            "<li style='font-size:11px'>%s</li>" % r for r in razones
        )

        sesgos = patrones.get("sesgos_detectados", [])
        sesgos_html = ""
        if patrones.get("disponible") and sesgos:
            sesgos_html = (
                "<br><b style='font-size:11px'>Sesgo sistematico detectado:</b>"
                "<br><span style='font-size:11px;color:%s'>%s</span>"
            ) % (ROJO, " | ".join(sesgos))

        html += (
            "<div style='background:#f9f9f9;border-left:4px solid %s;"
            "border-radius:4px;padding:10px;margin-bottom:10px'>"
            "<b style='color:%s'>%s</b><br>"
            "<b>Combinacion ganadora:</b> "
            "<span style='letter-spacing:2px;font-weight:bold'>%s</span>"
            " (suma: %d | score ML: %.4f)<br>"
            "<span style='color:%s'><b>%s</b></span><br>"
            "Rank estimado entre candidatos: <b>%s</b><br>"
            "<b style='font-size:11px'>Por que no quedo arriba:</b>"
            "<ul>%s</ul>%s"
            "</div>"
        ) % (
            color_sev(sev), AZUL, name,
            combo_str, suma, ml_score,
            top20_color, top20_txt,
            rank_txt,
            razones_html,
            sesgos_html,
        )

    html += (
        "<p style='font-size:10px;color:#aaa'>"
        "El aprendizaje retroactivo agrega las combinaciones ganadoras "
        "que el modelo fallo como ejemplos de entrenamiento reforzados. "
        "Con 3+ misses graves por juego, el modelo se reajusta automaticamente."
        "</p>"
    )
    return html