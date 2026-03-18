import sys
import os

# Agregar core/system/ al path para importar módulos locales
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jefe_maestro_v6_elite_predictor import run_model, send_email_results
from rocket_phoenix import run_rocket_phoenix, generar_html_aciertos


def main():
    # ── 1. Rocket Phoenix: calcular aciertos del sorteo anterior ──
    print("🔥 Ejecutando Rocket Phoenix...")
    try:
        resumen_aciertos = run_rocket_phoenix()
        html_aciertos    = generar_html_aciertos(resumen_aciertos)
        if resumen_aciertos:
            for name, data in resumen_aciertos.items():
                print(f"  [{name}] Mejor: {data.get('max_matches',0)}/6 | "
                      f"Promedio: {data.get('avg_matches',0):.2f}")
        else:
            print("  (Sin aciertos previos que reportar aún)")
    except Exception as e:
        print(f"  Rocket Phoenix error (no crítico): {e}")
        html_aciertos = ""

    # ── 2. Correr el modelo y obtener nuevas predicciones ─────────
    print("\n🚀 Corriendo Jefe Maestro...")
    df, stats = run_model(html_aciertos_extra=html_aciertos)

    print("\nModelo ejecutado correctamente.")
    print(df.head())
    print(stats)


if __name__ == "__main__":
    main()
