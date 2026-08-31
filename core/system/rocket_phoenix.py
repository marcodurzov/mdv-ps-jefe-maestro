#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rocket_phoenix.py  v2
Loop de autoaprendizaje del Jefe Maestro.
Mejora v2: muestra el ranking (#1-#20) de la mejor combinacion predicha.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Optional

try:
    from retroactive_learner import run_retroactive_learning, generar_html_retroactivo
    RETROACTIVE_AVAILABLE = True
except Exception:
    RETROACTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(os.path.dirname(_HERE))
_DATA_DIR = os.path.join(_ROOT, "data")
_CACHE    = os.path.join(_ROOT, "cache")

PREDICTIONS_HISTORY = os.path.join(_ROOT, "predictions_history_v8.json")
TRACKING_FILE       = os.path.join(_ROOT, "rocket_phoenix_tracking.json")
MODEL_FILE_TPL      = os.path.join(_CACHE, "model_v8_{name}.joblib")

LOTTERIES = {
    "Melate":     {"k": 6, "has_bono": True},
    "Revancha":   {"k": 6, "has_bono": False},
    "Revanchita": {"k": 6, "has_bono": False},
}

MIN_SORTEOS_PARA_APRENDER = 50


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


def _get_latest_results():
    results = {}
    for name in LOTTERIES:
        csv_path = os.path.join(_DATA_DIR, "%s.csv" % name.lower())
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["_fecha_dt"] = pd.to_datetime(
                df["FECHA"], dayfirst=True, errors="coerce"
            )
            df = df.sort_values("_fecha_dt", ascending=False)
            row = df.iloc[0]
            k   = LOTTERIES[name]["k"]
            nums = [int(row["N%d" % i]) for i in range(1, k+1)
                    if pd.notna(row.get("N%d" % i))]
            bono = None
            if LOTTERIES[name]["has_bono"] and "BONO" in df.columns:
                if pd.notna(row.get("BONO")):
                    bono = int(row["BONO"])
            results[name] = {
                "fecha":   str(row["FECHA"]),
                "numeros": sorted(nums),
                "bono":    bono,
            }
        except Exception as e:
            logger.warning("[%s] Error leyendo CSV: %s", name, e)
    return results


def _calcular_aciertos(predicciones, resultado):
    resultado_set = set(resultado)
    out = []
    for rank, pred in enumerate(predicciones, 1):
        combo   = set(pred.get("combo", []))
        matches = len(combo & resultado_set)
        out.append({
            "combo":   sorted(combo),
            "matches": matches,
            "score":   float(pred.get("global_composite", 0.0)),
            "rank":    rank,   # posicion en el top 20 original
        })
    return sorted(out, key=lambda x: x["matches"], reverse=True)


def _find_predictions_before(fecha_resultado, history):
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


def _reajustar_modelo(name, sorteos):
    try:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        return

    mf = MODEL_FILE_TPL.format(name=name)
    if not os.path.exists(mf):
        return

    sorteos_name = [s for s in sorteos
                    if s.get("lottery") == name and s.get("aciertos")]
    if len(sorteos_name) < MIN_SORTEOS_PARA_APRENDER:
        logger.info("[%s] Phoenix: %d/%d sorteos. Sin reajuste aun.",
                    name, len(sorteos_name), MIN_SORTEOS_PARA_APRENDER)
        return

    try:
        cached = joblib.load(mf)
        if cached.get("models") is None:
            return

        X_adj, y_adj = [], []
        for s in sorteos_name[-30:]:
            for ac in s.get("aciertos", []):
                sc = float(ac.get("score", 0.5))
                m  = int(ac.get("matches", 0))
                X_adj.append([sc, sc**2, float(m > 0), float(m >= 2)])
                y_adj.append(1 if m >= 3 else 0)

        if len(X_adj) < 20 or len(set(y_adj)) < 2:
            return

        X = np.array(X_adj, dtype=float)
        y = np.array(y_adj, dtype=int)

        meta_ph = HistGradientBoostingClassifier(max_iter=50, random_state=42)
        meta_ph.fit(X, y)
        cached["models"]["meta_phoenix"] = meta_ph
        cached["phoenix_updates"]        = cached.get("phoenix_updates", 0) + 1
        joblib.dump(cached, mf)
        logger.info("[%s] Phoenix: meta-learner reajustado (%d sorteos).",
                    name, len(sorteos_name))
    except Exception as e:
        logger.warning("[%s] Phoenix reajuste fallo: %s", name, e)


def run_rocket_phoenix():
    logger.info("Rocket Phoenix iniciando...")
    try:
        tracking   = _load_json(TRACKING_FILE, {"sorteos": []})
        history    = _load_json(PREDICTIONS_HISTORY, {})
        resultados = _get_latest_results()

        if not resultados:
            logger.warning("Phoenix: sin resultados reales.")
            return {}

        resumen        = {}
        nuevos_sorteos = []

        for name, resultado in resultados.items():
            fecha   = resultado["fecha"]
            numeros = resultado["numeros"]

            ya = any(s.get("lottery") == name and s.get("fecha") == fecha
                     for s in tracking.get("sorteos", []))
            if ya:
                for s in tracking.get("sorteos", []):
                    if s.get("lottery") == name and s.get("fecha") == fecha:
                        resumen[name] = {
                            "fecha":        fecha,
                            "numeros":      numeros,
                            "bono":         resultado.get("bono"),
                            "aciertos":     s.get("aciertos", []),
                            "mejor":        s.get("mejor_combo", {}),
                            "max_matches":  s.get("max_matches", 0),
                            "avg_matches":  s.get("avg_matches", 0.0),
                            "distribucion": s.get("distribucion", {}),
                        }
                continue

            preds = _find_predictions_before(fecha, history)
            if not preds:
                logger.info("[%s] Sin predicciones previas a %s.", name, fecha)
                continue

            aciertos = _calcular_aciertos(preds, numeros)
            max_m    = max((a["matches"] for a in aciertos), default=0)
            avg_m    = float(np.mean([a["matches"] for a in aciertos])) \
                       if aciertos else 0.0
            dist     = {i: sum(1 for a in aciertos if a["matches"] == i)
                        for i in range(7)}
            mejor    = aciertos[0] if aciertos else {}

            sorteo_data = {
                "lottery":        name,
                "fecha":          fecha,
                "resultado":      numeros,
                "bono":           resultado.get("bono"),
                "aciertos":       aciertos[:5],
                "mejor_combo":    mejor,
                "max_matches":    max_m,
                "avg_matches":    round(avg_m, 3),
                "distribucion":   dist,
                "n_predicciones": len(preds),
                "timestamp":      datetime.now().isoformat(),
            }
            nuevos_sorteos.append(sorteo_data)
            resumen[name] = {
                "fecha":        fecha,
                "numeros":      numeros,
                "bono":         resultado.get("bono"),
                "aciertos":     aciertos[:5],
                "mejor":        mejor,
                "max_matches":  max_m,
                "avg_matches":  round(avg_m, 3),
                "distribucion": dist,
            }
            logger.info("[%s] %s mejor: %d/6 promedio: %.2f",
                        name, fecha, max_m, avg_m)

        if nuevos_sorteos:
            tracking.setdefault("sorteos", []).extend(nuevos_sorteos)
            tracking["sorteos"]       = tracking["sorteos"][-100:]
            tracking["ultimo_update"] = datetime.now().isoformat()
            _save_json(TRACKING_FILE, tracking)
            logger.info("Phoenix: %d sorteos nuevos guardados.", len(nuevos_sorteos))
            for name in LOTTERIES:
                _reajustar_modelo(name, tracking["sorteos"])

        logger.info("Rocket Phoenix completado.")
        return resumen

    except Exception as e:
        logger.error("Rocket Phoenix fallo: %s", e)
        return {}


def generar_html_aciertos(resumen):
    if not resumen:
        return ""

    AZUL    = "#1a3a5c"
    VERDE   = "#2e7d32"
    NARANJA = "#e65100"
    GRIS    = "#666666"

    def color_m(m):
        if m >= 4: return VERDE
        if m >= 2: return NARANJA
        return GRIS

    html = (
        "<h3 style='color:%s;border-bottom:2px solid #2e75b6;"
        "padding-bottom:6px'>Reporte de Aciertos - Sorteo Anterior</h3>" % AZUL
    )

    for name, data in resumen.items():
        fecha    = data.get("fecha", "")
        numeros  = data.get("numeros", [])
        bono     = data.get("bono")
        max_m    = data.get("max_matches", 0)
        avg_m    = data.get("avg_matches", 0.0)
        mejor    = data.get("mejor", {})
        dist     = data.get("distribucion", {})

        nums_str  = " ".join("%02d" % n for n in numeros)
        bono_str  = " + BONO <b>%02d</b>" % bono if bono else ""
        mejor_str = " ".join("%02d" % n for n in mejor.get("combo", []))
        mejor_m   = mejor.get("matches", 0)
        mejor_rank = mejor.get("rank", "?")  # NUEVO: posicion en el ranking
        mejor_score = mejor.get("score", 0.0)

        dist_html = ""
        for i in range(6, -1, -1):
            n = dist.get(i, 0)
            if n > 0:
                bar = "█" * n
                dist_html += (
                    "&nbsp;&nbsp;%d aciertos: "
                    "<span style='color:%s'>%s</span> (%d combos)<br>"
                ) % (i, color_m(i), bar, n)

        html += (
            "<div style='background:#f9f9f9;border-left:4px solid %s;"
            "border-radius:4px;padding:12px;margin-bottom:12px'>"
            "<b style='color:%s;font-size:14px'>%s</b>"
            " &nbsp;·&nbsp; Sorteo del %s<br><br>"
            "<b>Resultado oficial:</b>&nbsp;"
            "<span style='font-size:15px;letter-spacing:2px;font-weight:bold'>"
            "%s</span>%s<br><br>"
            "<table style='width:100%%;font-size:13px'><tr>"
            "<td><b>Mejor combo predicho:</b><br>"
            "<span style='letter-spacing:1px;font-weight:bold'>%s</span><br>"
            "<span style='color:%s;font-weight:bold;font-size:16px'>%d/6</span>"
            " aciertos"
            "<br><span style='font-size:11px;color:#888'>"
            "Posicion en ranking: <b>#%s</b> | Score: %.5f</span>"
            "</td>"
            "<td style='text-align:right;vertical-align:top'>"
            "<b>Promedio top 20:</b><br>"
            "<span style='font-size:20px;font-weight:bold;color:%s'>%.2f</span><br>"
            "<span style='font-size:11px;color:#888'>aciertos / combo</span>"
            "</td></tr></table><br>"
            "<b style='font-size:12px'>Distribucion:</b><br>"
            "<span style='font-size:12px'>%s</span>"
            "</div>"
        ) % (
            color_m(max_m), AZUL, name, fecha,
            nums_str, bono_str,
            mejor_str, color_m(mejor_m), mejor_m,
            mejor_rank, mejor_score,
            color_m(int(avg_m)), avg_m,
            dist_html
        )

    html += (
        "<p style='font-size:11px;color:#888'>"
        "Rocket Phoenix aprende de estos aciertos. "
        "Con %d+ sorteos trackeados, el modelo se reajusta automaticamente."
        "</p>" % MIN_SORTEOS_PARA_APRENDER
    )
    return html