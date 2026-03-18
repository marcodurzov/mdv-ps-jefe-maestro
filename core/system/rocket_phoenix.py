#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rocket_phoenix.py
Loop de autoaprendizaje del Jefe Maestro.

FUNCIONAMIENTO:
  1. Lee el resultado real más reciente de cada sorteo (del CSV).
  2. Busca en predictions_history_v8.json las predicciones hechas
     ANTES de ese sorteo.
  3. Calcula cuántos números acertó cada combinación predicha (0-6).
  4. Guarda el tracking en rocket_phoenix_tracking.json.
  5. Después de MIN_SORTEOS_PARA_APRENDER sorteos trackeados, ajusta
     los pesos del meta-learner usando los aciertos como señal.
  6. Retorna un resumen HTML de aciertos para incluir en el correo.

INTEGRACIÓN:
  - Se llama desde main_run.py antes de enviar el correo.
  - Completamente aislado: si falla, no interrumpe el pipeline.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Rutas ────────────────────────────────────────────────────────────
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

MIN_SORTEOS_PARA_APRENDER = 10


# ─────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────

def _safe_json(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj

def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def _save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_safe_json)
    except Exception as e:
        logger.error(f"Error guardando {path}: {e}")


# ─────────────────────────────────────────────────────────────────────
# LECTURA DE RESULTADOS REALES
# ─────────────────────────────────────────────────────────────────────

def _get_latest_results() -> Dict[str, Dict]:
    results = {}
    for name in LOTTERIES:
        csv_path = os.path.join(_DATA_DIR, f"{name.lower()}.csv")
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
            nums = [int(row[f"N{i}"]) for i in range(1, k+1)
                    if pd.notna(row.get(f"N{i}"))]
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
            logger.warning(f"[{name}] Error leyendo CSV: {e}")
    return results


# ─────────────────────────────────────────────────────────────────────
# CÁLCULO DE ACIERTOS
# ─────────────────────────────────────────────────────────────────────

def _calcular_aciertos(predicciones: List[Dict],
                        resultado: List[int]) -> List[Dict]:
    resultado_set = set(resultado)
    out = []
    for pred in predicciones:
        combo   = set(pred.get("combo", []))
        matches = len(combo & resultado_set)
        out.append({
            "combo":   sorted(combo),
            "matches": matches,
            "score":   float(pred.get("global_composite", 0.0)),
        })
    return sorted(out, key=lambda x: x["matches"], reverse=True)


def _find_predictions_before(fecha_resultado: str,
                               history: Dict) -> Optional[List[Dict]]:
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
# REAJUSTE DEL META-LEARNER
# ─────────────────────────────────────────────────────────────────────

def _reajustar_modelo(name: str, sorteos: List[Dict]):
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
        logger.info(f"[{name}] Phoenix: {len(sorteos_name)}/"
                    f"{MIN_SORTEOS_PARA_APRENDER} sorteos. Sin reajuste aún.")
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
        logger.info(f"[{name}] Phoenix: meta-learner reajustado "
                    f"({len(sorteos_name)} sorteos).")
    except Exception as e:
        logger.warning(f"[{name}] Phoenix reajuste falló: {e}")


# ─────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

def run_rocket_phoenix() -> Dict:
    """
    Ejecuta el loop de autoaprendizaje.
    Retorna dict con resumen de aciertos para el correo.
    """
    logger.info("🔥 Rocket Phoenix — iniciando...")
    try:
        tracking   = _load_json(TRACKING_FILE, {"sorteos": []})
        history    = _load_json(PREDICTIONS_HISTORY, {})
        resultados = _get_latest_results()

        if not resultados:
            logger.warning("Phoenix: sin resultados reales disponibles.")
            return {}

        resumen:        Dict[str, Dict] = {}
        nuevos_sorteos: List[Dict]      = []

        for name, resultado in resultados.items():
            fecha   = resultado["fecha"]
            numeros = resultado["numeros"]

            # ¿Ya trackeado?
            ya = any(s.get("lottery") == name and s.get("fecha") == fecha
                     for s in tracking.get("sorteos", []))
            if ya:
                for s in tracking.get("sorteos", []):
                    if s.get("lottery") == name and s.get("fecha") == fecha:
                        resumen[name] = {
                            "fecha":       fecha,
                            "numeros":     numeros,
                            "bono":        resultado.get("bono"),
                            "aciertos":    s.get("aciertos", []),
                            "mejor":       s.get("mejor_combo", {}),
                            "max_matches": s.get("max_matches", 0),
                            "avg_matches": s.get("avg_matches", 0.0),
                            "distribucion": s.get("distribucion", {}),
                        }
                continue

            preds = _find_predictions_before(fecha, history)
            if not preds:
                logger.info(f"[{name}] Sin predicciones previas a {fecha}.")
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
            logger.info(f"[{name}] {fecha} — mejor: {max_m}/6 | "
                        f"promedio: {avg_m:.2f} | {numeros}")

        if nuevos_sorteos:
            tracking.setdefault("sorteos", []).extend(nuevos_sorteos)
            tracking["sorteos"]        = tracking["sorteos"][-100:]
            tracking["ultimo_update"]  = datetime.now().isoformat()
            _save_json(TRACKING_FILE, tracking)
            logger.info(f"Phoenix: {len(nuevos_sorteos)} sorteos nuevos guardados.")
            for name in LOTTERIES:
                _reajustar_modelo(name, tracking["sorteos"])

        logger.info("✅ Rocket Phoenix completado.")
        return resumen

    except Exception as e:
        logger.error(f"Rocket Phoenix falló (pipeline continúa): {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────
# HTML PARA EL CORREO
# ─────────────────────────────────────────────────────────────────────

def generar_html_aciertos(resumen: Dict) -> str:
    if not resumen:
        return ""

    AZUL    = "#1a3a5c"
    VERDE   = "#2e7d32"
    NARANJA = "#e65100"
    GRIS    = "#666666"

    def color_m(m: int) -> str:
        if m >= 4: return VERDE
        if m >= 2: return NARANJA
        return GRIS

    html = (f"<h3 style='color:{AZUL};border-bottom:2px solid #2e75b6;"
            f"padding-bottom:6px'>🎯 Reporte de Aciertos — Sorteo Anterior</h3>")

    for name, data in resumen.items():
        fecha    = data.get("fecha", "")
        numeros  = data.get("numeros", [])
        bono     = data.get("bono")
        max_m    = data.get("max_matches", 0)
        avg_m    = data.get("avg_matches", 0.0)
        mejor    = data.get("mejor", {})
        dist     = data.get("distribucion", {})

        nums_str  = " ".join(f"{n:02d}" for n in numeros)
        bono_str  = f" + BONO <b>{bono:02d}</b>" if bono else ""
        mejor_str = " ".join(f"{n:02d}" for n in mejor.get("combo", []))
        mejor_m   = mejor.get("matches", 0)

        dist_html = ""
        for i in range(6, -1, -1):
            n = dist.get(i, 0)
            if n > 0:
                bar = "█" * n
                dist_html += (f"&nbsp;&nbsp;{i} aciertos: "
                              f"<span style='color:{color_m(i)}'>{bar}</span>"
                              f" ({n} combos)<br>")

        html += f"""
<div style='background:#f9f9f9;border-left:4px solid {color_m(max_m)};
     border-radius:4px;padding:12px;margin-bottom:12px'>
  <b style='color:{AZUL};font-size:14px'>{name}</b>
  &nbsp;·&nbsp;Sorteo del {fecha}<br><br>
  <b>Resultado oficial:</b>&nbsp;
  <span style='font-size:15px;letter-spacing:2px;font-weight:bold'>
    {nums_str}</span>{bono_str}<br><br>
  <table style='width:100%;font-size:13px'>
    <tr>
      <td>
        <b>Mejor combo predicho:</b><br>
        <span style='letter-spacing:1px;font-weight:bold'>{mejor_str}</span><br>
        <span style='color:{color_m(mejor_m)};font-weight:bold;font-size:16px'>
          {mejor_m}/6</span> aciertos
      </td>
      <td style='text-align:right;vertical-align:top'>
        <b>Promedio top 20:</b><br>
        <span style='font-size:20px;font-weight:bold;color:{color_m(int(avg_m))}'>
          {avg_m:.2f}</span><br>
        <span style='font-size:11px;color:#888'>aciertos / combo</span>
      </td>
    </tr>
  </table>
  <br>
  <b style='font-size:12px'>Distribución:</b><br>
  <span style='font-size:12px'>{dist_html}</span>
</div>"""

    html += (f"<p style='font-size:11px;color:#888'>"
             f"🔥 Rocket Phoenix aprende de estos aciertos. "
             f"Con {MIN_SORTEOS_PARA_APRENDER}+ sorteos trackeados, "
             f"el modelo se reajusta automáticamente.</p>")
    return html
