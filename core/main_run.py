from system.database import load_history
from jefe_maestro_v6_elite_predictor import main as run_model
from system.mailer import send_email

if __name__=="__main__":
    histories={
        "Melate":load_history("Melate"),
        "Revancha":load_history("Revancha"),
        "Revanchita":load_history("Revanchita")
    }

    df,stats=run_model(histories_override=histories)
    send_email(df,stats)
def main():
    print("Iniciando Jefe Maestro Elite Predictor...")
    
    # Si tu modelo ya tiene una función principal diferente,
    # reemplaza "run()" por el nombre real.
    
    try:
        run()
    except NameError:
        print("No existe función run(). Ajustar nombre de función principal.")
