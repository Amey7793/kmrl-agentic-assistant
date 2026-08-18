"""
===============================================================================
MODULE: GTFS Database Ingestion Script (build_gtfs_db.py)
PURPOSE:
  Ingests 11 raw General Transit Feed Specification (GTFS) CSV/text files from 
  'Knowledge Base/OpenData/' and parses them into a relational SQLite 3 database ('gtfs.db').

DATA TABLES PROCESSED:
  - stops: Station details, GPS coordinates, stop IDs, and names.
  - routes: Train route definitions (Line 1).
  - trips: Individual train trip definitions and direction IDs.
  - stop_times: Exact arrival and departure schedules for every station stop.
  - fare_attributes: Price slabs and currency rules (F1 to F6).
  - fare_rules: Origin to destination station fare mapping rules.
===============================================================================
"""

import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENDATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "Knowledge Base", "OpenData")
DB_PATH = os.path.join(BASE_DIR, "gtfs.db")

def build_gtfs_database():
    print(f"Building GTFS SQLite database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Load stops.txt
    stops_file = os.path.join(OPENDATA_DIR, "stops.txt")
    if os.path.exists(stops_file):
        df_stops = pd.read_csv(stops_file)
        df_stops.to_sql("stops", conn, if_exists="replace", index=False)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stops_id ON stops(stop_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stops_name ON stops(stop_name);")
        print(f"Loaded {len(df_stops)} stops.")

    # Load routes.txt
    routes_file = os.path.join(OPENDATA_DIR, "routes.txt")
    if os.path.exists(routes_file):
        df_routes = pd.read_csv(routes_file)
        df_routes.to_sql("routes", conn, if_exists="replace", index=False)
        print(f"Loaded {len(df_routes)} routes.")

    # Load trips.txt
    trips_file = os.path.join(OPENDATA_DIR, "trips.txt")
    if os.path.exists(trips_file):
        df_trips = pd.read_csv(trips_file)
        df_trips.to_sql("trips", conn, if_exists="replace", index=False)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trips_trip_id ON trips(trip_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trips_route_id ON trips(route_id);")
        print(f"Loaded {len(df_trips)} trips.")

    # Load stop_times.txt
    stop_times_file = os.path.join(OPENDATA_DIR, "stop_times.txt")
    if os.path.exists(stop_times_file):
        df_stop_times = pd.read_csv(stop_times_file)
        df_stop_times.to_sql("stop_times", conn, if_exists="replace", index=False)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_stop ON stop_times(stop_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stop_times_trip ON stop_times(trip_id);")
        print(f"Loaded {len(df_stop_times)} stop_times rows.")

    # Load fare_attributes.txt
    fare_attr_file = os.path.join(OPENDATA_DIR, "fare_attributes.txt")
    if os.path.exists(fare_attr_file):
        df_fare_attr = pd.read_csv(fare_attr_file)
        df_fare_attr.to_sql("fare_attributes", conn, if_exists="replace", index=False)
        print(f"Loaded {len(df_fare_attr)} fare_attributes.")

    # Load fare_rules.txt
    fare_rules_file = os.path.join(OPENDATA_DIR, "fare_rules.txt")
    if os.path.exists(fare_rules_file):
        df_fare_rules = pd.read_csv(fare_rules_file)
        df_fare_rules.to_sql("fare_rules", conn, if_exists="replace", index=False)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fare_rules_orig_dest ON fare_rules(origin_id, destination_id);")
        print(f"Loaded {len(df_fare_rules)} fare_rules.")

    conn.commit()
    conn.close()
    print("GTFS database build complete!")

if __name__ == "__main__":
    build_gtfs_database()
