from system.database import initialize_database, load_history
from model_runner import run_model  # asegúrate que este import sea el correcto según tu proyecto

def main():

    # Inicializar base SQLite
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
