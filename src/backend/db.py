from typing import Dict, List
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
    cur.execute(
        f"""
        SELECT Name, Description, Brand, Category, Color, Gender, Size
        FROM {TABLE_NAME} WHERE Pid = %s
    """,
        (pid,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        return {
            "Name": row[0],
            "Description": row[1],
            "Brand": row[2],
            "Category": row[3],
            "Color": row[4],
            "Gender": row[5],
            "Size": row[6],
        }
    return None


def fetch_products_by_pids(pids: List, scores: List) -> list[Dict]:
    """Fetches multiple produc given the ids

    Parameters
    ----------
    pid : str
        A list of pids passed in

    Returns
    -------
    _type_
        A list of Dictionaries
    """
    conn = get_db_connection()
    cur = conn.cursor()

    placeholders = ", ".join(["%s"] * len(pids))
    query = f"""
        SELECT Pid, Name, Description, Brand, Category, Color, Gender, Size
        FROM {TABLE_NAME} WHERE Pid IN ({placeholders})
    """
    cur.execute(query, tuple(pids))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Map Pid to full product details
    pid_to_details = {
        row[0]: {
            "Pid": row[0],
            "Name": row[1],
            "Description": row[2],
            "Brand": row[3],
            "Category": row[4],
            "Color": row[5],
            "Gender": row[6],
            "Size": row[7],
        }
        for row in rows
    }

    # Sort PIDs by the given scores
    sorted_pid_score_pairs = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)

    # Build final sorted result using list comprehension and include score
    results = [
        {**pid_to_details[pid], "similarity": score}
        for pid, score in sorted_pid_score_pairs
        if pid in pid_to_details
    ]

    return results
