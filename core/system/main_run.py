import sys
import os

# Agregar la carpeta core/system/ al path para importar el predictor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jefe_maestro_v6_elite_predictor import run_model


def main():
    # Carga directamente desde los CSV (fuente de verdad)
    df, stats = run_model()

    print("Modelo ejecutado correctamente.")
    print(df.head())
    print(stats)


if __name__ == "__main__":
    main()
