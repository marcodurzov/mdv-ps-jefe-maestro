from system.jefe_maestro_v6_elite_predictor import run_model


def main():
    # Cargamos directamente desde los CSV (fuente de verdad).
    # La base de datos queda como respaldo opcional pero no se usa
    # en el pipeline principal para evitar datos sucios.
    df, stats = run_model()

    print("Modelo ejecutado correctamente.")
    print(df.head())
    print(stats)


if __name__ == "__main__":
    main()
