"""
===============================================================================
MODULE: Lost & Found Database Ingestion Script (build_lost_found_db.py)
PURPOSE:
  Parses the 3.7MB 'Lost and Found.json' database (~31,626 items) into an 
  indexed SQLite 3 database ('lost_found.db') featuring an FTS5 virtual table
  for sub-millisecond full-text search across categories, dates, stations, and LFIDs.
===============================================================================
"""

import os
import json
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(os.path.dirname(BASE_DIR), "Knowledge Base", "JSONs", "Lost and Found.json")
DB_PATH = os.path.join(BASE_DIR, "lost_found.db")

def build_lost_found_database():
    print(f"Building Lost & Found SQLite database at: {DB_PATH}")
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found!")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for sublist in data.get("rows", []):
        for i in range(0, len(sublist), 6):
            rec = sublist[i:i+6]
            if len(rec) == 6:
                items.append((
                    rec[0].strip(), # sl_no
                    rec[1].strip(), # found_date
                    rec[2].strip(), # category
                    rec[3].strip(), # found_station
                    rec[4].strip(), # current_station
                    rec[5].strip()  # lfid
                ))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS lost_items;")
    cursor.execute("""
        CREATE TABLE lost_items (
            sl_no TEXT,
            found_date TEXT,
            category TEXT,
            found_station TEXT,
            current_station TEXT,
            lfid TEXT PRIMARY KEY
        );
    """)

    cursor.executemany("""
        INSERT OR REPLACE INTO lost_items (sl_no, found_date, category, found_station, current_station, lfid)
        VALUES (?, ?, ?, ?, ?, ?);
    """, items)

    # Build FTS5 virtual table for full-text search across category, stations, and lfid
    cursor.execute("DROP TABLE IF EXISTS lost_items_fts;")
    cursor.execute("""
        CREATE VIRTUAL TABLE lost_items_fts USING fts5(
            category,
            found_station,
            current_station,
            lfid,
            found_date,
            content='lost_items',
            content_rowid='rowid'
        );
    """)

    cursor.execute("""
        INSERT INTO lost_items_fts(rowid, category, found_station, current_station, lfid, found_date)
        SELECT rowid, category, found_station, current_station, lfid, found_date FROM lost_items;
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lf_date ON lost_items(found_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lf_category ON lost_items(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lf_found_station ON lost_items(found_station);")

    conn.commit()
    conn.close()
    print(f"Lost & Found database build complete! Indexed {len(items)} records.")

if __name__ == "__main__":
    build_lost_found_database()
