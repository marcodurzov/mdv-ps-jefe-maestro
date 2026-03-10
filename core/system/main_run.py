from system.database import initialize_database, load_history
from jefe_maestro_v6_elite_predictor import run_model


def main():

    initialize_database()

    histories = {
        "Melate": load_history("Melate"),
        "Revancha": load_history("Revancha"),
        "Revanchita": load_history("Revanchita")
    }

    df, stats = run_model(histories_override=histories)

    print("Modelo ejecutado correctamente.")
    print(df.head())
    print(stats)


if __name__ == "__main__":
    main()
