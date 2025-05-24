import psycopg2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.db import DB_CONFIG, TABLE_NAME

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def fetch_product_by_pid(pid):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT Name, Description, Brand, Category, Color, Gender, Size, Price
        FROM {TABLE_NAME} WHERE Pid = %s
    """, (pid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    
    if row:
        return {
            'Name': row[0],
            'Description': row[1],
            'Brand': row[2],
            'Category': row[3],
            'Color': row[4],
            'Gender': row[5],
            'Size': row[6],
            'Price': row[7],
        }
    return None
