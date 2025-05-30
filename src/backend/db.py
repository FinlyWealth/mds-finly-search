import psycopg2
import os
import sys
from typing import Dict, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import config.db


def get_db_connection():
    return psycopg2.connect(**config.db.DB_CONFIG)


def fetch_products_by_pids(pids):
    """Fetch multiple products by their PIDs in a single database query.

    Args:
        pids: List of product IDs to fetch

    Returns:
        Dict mapping PIDs to product details
    """
    if not pids:
        return {}

    conn = get_db_connection()
    cur = conn.cursor()

    # Convert list of PIDs to a comma-separated string for SQL
    pid_list = ",".join([f"'{pid}'" for pid in pids])

    cur.execute(
        f"""
        SELECT Pid, Name, Description, Brand, Category, Color, Gender, Size, Price
        FROM {config.db.TABLE_NAME} WHERE Pid IN ({pid_list})
    """
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Convert rows to dictionary mapping PIDs to product details
    products = {}
    for row in rows:
        products[row[0]] = {
            "Name": row[1],
            "Description": row[2],
            "Brand": row[3],
            "Category": row[4],
            "Color": row[5],
            "Gender": row[6],
            "Size": row[7],
            "Price": row[8],
        }

    return products
