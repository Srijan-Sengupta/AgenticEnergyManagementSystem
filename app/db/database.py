import os
import sqlite3
import pandas as pd
from app.core.config import DB_PATH

def setup_database(csv_path: str, db_path: str = DB_PATH):
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    conn = sqlite3.connect(db_path)
    df.to_sql('demand_reports', conn, if_exists='replace', index=False)
    schema = pd.io.sql.get_schema(df, 'demand_reports')
    conn.close()
    return schema

def get_dynamic_schema(db_path: str = DB_PATH) -> str:
    try:
        if not os.path.exists(db_path):
            return "No tables exist in the database yet."
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        schemas = [row[0] for row in cursor.fetchall() if row[0] is not None]
        conn.close()
        return "\n\n".join(schemas) if schemas else "No tables exist in the database yet."
    except Exception as e:
        return f"Error reading schema: {e}"