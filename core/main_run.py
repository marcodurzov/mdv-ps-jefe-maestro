from system.database import initialize_database, load_history
from system.database import initialize_database, load_history
from core.jefe_maestro_v6_elite_predictor import run_model


def main():

    # Inicializar base de datos
    initialize_database()

    # Cargar históricos desde SQLite
    histories = {
        "Melate": load_history("Melate"),
        "Revancha": load_history("Revancha"),
        "Revanchita": load_history("Revanchita")
    }

    # Ejecutar modelo
    df, stats = run_model(histories_override=histories)

    print("Modelo ejecutado correctamente.")
    print(df.head())
    print(stats)


if __name__ == "__main__":
    main()

