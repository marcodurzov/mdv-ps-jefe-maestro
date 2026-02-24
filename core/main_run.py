from system.database import initialize_database
from system.database import load_history

initialize_database()

data = {
    "Melate": load_history("Melate"),
    "Revancha": load_history("Revancha"),
    "Revanchita": load_history("Revanchita")
}

    df,stats=run_model(histories_override=histories)
def main():
    print("Iniciando Jefe Maestro Elite Predictor...")
    
    # Si tu modelo ya tiene una función principal diferente,
    # reemplaza "run()" por el nombre real.
    
    try:
        run()
    except NameError:
        print("No existe función run(). Ajustar nombre de función principal.")
