import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jefe_maestro_v6_elite_predictor import run_model, send_email_results
from rocket_phoenix import run_rocket_phoenix, generar_html_aciertos

# Retroactive learner: sin importacion circular
try:
    from retroactive_learner import run_retroactive_learning, generar_html_retroactivo
    RETROACTIVE_OK = True
except Exception as e:
    print("Retroactive learner no disponible: %s" % e)
    RETROACTIVE_OK = False

try:
    from auto_tuner import run_auto_tuner, generar_html_tuner
    AUTO_TUNER_OK = True
except Exception as e:
    print("Auto-Tuner no disponible: %s" % e)
    AUTO_TUNER_OK = False


def main():
    # ── 1. Rocket Phoenix: aciertos del sorteo anterior ──
    print("Rocket Phoenix iniciando...")
    try:
        resumen_aciertos = run_rocket_phoenix()
        html_aciertos    = generar_html_aciertos(resumen_aciertos)
        for name, data in resumen_aciertos.items():
            print("  [%s] Mejor: %d/6 | Promedio: %.2f" % (
                name, data.get("max_matches", 0), data.get("avg_matches", 0)
            ))
    except Exception as e:
        print("Rocket Phoenix error: %s" % e)
        html_aciertos = ""

    # ── 2. Retroactive Learning: por que no quedo el ganador arriba ──
    html_retro = ""
    if RETROACTIVE_OK:
        print("\nRetroactive Learner iniciando...")
        try:
            # Cargar historiales para el learner
            import pandas as pd
            _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _DATA = os.path.join(_ROOT, "data")

            all_histories = {}
            all_stats     = {}
            for game in ["Melate", "Revancha", "Revanchita"]:
                path = os.path.join(_DATA, "%s.csv" % game.lower())
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df["_fecha_dt"] = pd.to_datetime(
                        df["FECHA"], dayfirst=True, errors="coerce"
                    )
                    df = df.sort_values("_fecha_dt", ascending=False).reset_index(drop=True)
                    all_histories[game] = df

            # Cargar stats desde cache si existen
            import joblib
            _CACHE = os.path.join(_ROOT, "cache")
            for game in all_histories:
                sf = os.path.join(_CACHE, "stats_v8_%s.joblib" % game)
                if os.path.exists(sf):
                    try:
                        all_stats[game] = joblib.load(sf)
                    except Exception:
                        all_stats[game] = {}
                else:
                    all_stats[game] = {}

            if all_histories:
                retro = run_retroactive_learning(all_histories, all_stats)
                html_retro = generar_html_retroactivo(retro)
                for name, data in retro.items():
                    print("  [%s] Ganador: %s | En top20: %s | Severidad: %s" % (
                        name,
                        data.get("winning_combo", []),
                        data.get("estuvo_en_top20", False),
                        data.get("severidad", "?")
                    ))
        except Exception as e:
            print("Retroactive Learner error (no critico): %s" % e)

    # ── 3. Auto-Tuner: optimizar hiperparametros con evidencia real ──
    html_tuner = ""
    if AUTO_TUNER_OK:
        print("\nAuto-Tuner evaluando configuracion...")
        try:
            resultado_tuner = run_auto_tuner()
            html_tuner = generar_html_tuner(resultado_tuner)
            if resultado_tuner.get("ejecutado"):
                print("  Aplicado: %s | Mejora: %.1f%%" % (
                    resultado_tuner.get("aplicado"),
                    resultado_tuner.get("mejora_pct", 0)
                ))
            else:
                print("  %s" % resultado_tuner.get("razon", ""))
        except Exception as e:
            print("Auto-Tuner error (no critico): %s" % e)

    # ── 4. Jefe Maestro: nuevas predicciones ──
    print("\nJefe Maestro iniciando...")
    df, stats = run_model(
        html_aciertos_extra=html_aciertos + html_retro + html_tuner
    )

    print("\nModelo ejecutado correctamente.")
    print(df.head())
    print(stats)


if __name__ == "__main__":
    main()