import psycopg2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.db import DB_CONFIG

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def fetch_product_by_pid(pid):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT Name, Description, Brand, Manufacturer, Color, Gender, Size
        FROM products WHERE Pid = %s
    """, (pid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if row:
        return {
            'Name': row[0],
            'Description': row[1],
            'Brand': row[2],
            'Manufacturer': row[3],
            'Color': row[4],
            'Gender': row[5],
            'Size': row[6]
        }
    return None
