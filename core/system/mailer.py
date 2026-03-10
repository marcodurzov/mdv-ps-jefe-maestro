import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

def send_email(df, stats):
    user = os.getenv("EMAIL_USER")
    pw = os.getenv("EMAIL_PASS")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"MDV PS - Resultados {datetime.now()}"
    msg["From"] = user
    msg["To"] = "marco.durzo@gmail.com"

    html = "<h2>Top Combinaciones</h2><table border='1'>"
    html += "<tr><th>#</th><th>Combinación</th><th>Score</th><th>Suma</th></tr>"

    for i,row in df.iterrows():
        combo=" - ".join(f"{n:02d}" for n in row["combo"])
        html+=f"<tr><td>{i+1}</td><td>{combo}</td><td>{row['global_composite']:.6f}</td><td>{row['suma']}</td></tr>"

    html+="</table><pre>"+stats+"</pre>"

    msg.attach(MIMEText(html,"html"))

    with smtplib.SMTP("smtp.gmail.com",587) as server:
        server.starttls()
        server.login(user,pw)
        server.send_message(msg)
