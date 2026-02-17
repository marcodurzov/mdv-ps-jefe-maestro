import sqlite3
import os

DB_PATH = os.path.join("data", "melate.db")

def get_connection():
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def initialize_database():
    conn = get_connection()
    cur = conn.cursor()

    for table in ["Melate", "Revancha", "Revanchita"]:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            n1 INTEGER NOT NULL,
            n2 INTEGER NOT NULL,
            n3 INTEGER NOT NULL,
            n4 INTEGER NOT NULL,
            n5 INTEGER NOT NULL,
            n6 INTEGER NOT NULL,
            UNIQUE(fecha, n1, n2, n3, n4, n5, n6)
        )
        """)

    conn.commit()
    conn.close()

def insert_draw(table, fecha, nums):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(f"""
        INSERT OR IGNORE INTO {table}
        (fecha,n1,n2,n3,n4,n5,n6)
        VALUES(?,?,?,?,?,?,?)
    """, (fecha,*nums))
    conn.commit()
    conn.close()

def load_history(table):
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query(
        f"SELECT fecha as FECHA,n1 as N1,n2 as N2,n3 as N3,n4 as N4,n5 as N5,n6 as N6 FROM {table} ORDER BY fecha ASC",
        conn
    )
    conn.close()
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    return df
