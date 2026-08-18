"""
===============================================================================
TOOL MODULE: Deterministic GTFS Transit Engine (gtfs_tool.py)
PURPOSE:
  Provides exact, non-hallucinated transit utilities by executing SQL joins and 
  NetworkX graph calculations against the SQLite database ('gtfs.db').

KEY FUNCTIONS:
  1. get_fare(origin, destination): Retrieves exact ticket prices (e.g. Rs. 60).
  2. find_route(origin, destination): Calculates station sequence, hop count, and travel time.
  3. get_next_trains(station, time): Retrieves upcoming scheduled train departures.
  4. get_all_stations(): Returns ordered list of all metro stations.
===============================================================================
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "gtfs.db")

class GTFSTool:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _fuzzy_match_station(self, conn, search_term):
        cursor = conn.cursor()
        search_term_clean = search_term.strip().lower()
        
        cursor.execute("SELECT stop_id, stop_name FROM stops;")
        rows = cursor.fetchall()
        
        # 1. Exact match
        for sid, sname in rows:
            if sname.lower() == search_term_clean:
                return sid, sname

        # 2. Substring match
        for sid, sname in rows:
            if search_term_clean in sname.lower() or sname.lower() in search_term_clean:
                return sid, sname

        # 3. Partial word match
        for sid, sname in rows:
            for word in search_term_clean.split():
                if len(word) > 2 and word in sname.lower():
                    return sid, sname

        return None, None

    def get_all_stations(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT stop_id, stop_name FROM stops ORDER BY stop_name;")
        stations = [{"stop_id": r[0], "stop_name": r[1]} for r in cursor.fetchall()]
        conn.close()
        return stations

    def get_fare(self, origin: str, destination: str):
        conn = self._get_connection()
        orig_id, orig_name = self._fuzzy_match_station(conn, origin)
        dest_id, dest_name = self._fuzzy_match_station(conn, destination)

        if not orig_id:
            conn.close()
            return {"error": f"Origin station '{origin}' not found."}
        if not dest_id:
            conn.close()
            return {"error": f"Destination station '{destination}' not found."}

        cursor = conn.cursor()
        query = """
            SELECT fa.fare_id, fa.price, fa.currency_type
            FROM fare_rules fr
            JOIN fare_attributes fa ON fr.fare_id = fa.fare_id
            WHERE fr.origin_id = ? AND fr.destination_id = ?;
        """
        cursor.execute(query, (orig_id, dest_id))
        row = cursor.fetchone()

        if not row:
            cursor.execute(query, (dest_id, orig_id))
            row = cursor.fetchone()

        conn.close()

        if row:
            fare_id, price, currency = row
            return {
                "origin_station": orig_name,
                "origin_id": orig_id,
                "destination_station": dest_name,
                "destination_id": dest_id,
                "fare_id": fare_id,
                "price": price,
                "currency": currency,
                "formatted": f"Ticket fare from {orig_name} to {dest_name} is Rs. {price:.0f}."
            }
        else:
            return {"error": f"No fare rule found between {orig_name} and {dest_name}."}

    def find_route(self, origin: str, destination: str):
        conn = self._get_connection()
        orig_id, orig_name = self._fuzzy_match_station(conn, origin)
        dest_id, dest_name = self._fuzzy_match_station(conn, destination)

        if not orig_id or not dest_id:
            conn.close()
            return {"error": f"Could not resolve stations: origin='{origin}', destination='{destination}'"}

        cursor = conn.cursor()
        # Retrieve the trip that visits the maximum number of stations to get the full line sequence
        query = """
            SELECT st.trip_id, COUNT(st.stop_sequence) as cnt
            FROM stop_times st
            GROUP BY st.trip_id
            ORDER BY cnt DESC
            LIMIT 1;
        """
        cursor.execute(query)
        best_trip_row = cursor.fetchone()
        best_trip_id = best_trip_row[0] if best_trip_row else 'WE_1'

        query_stops = """
            SELECT st.stop_sequence, s.stop_id, s.stop_name
            FROM stop_times st
            JOIN stops s ON st.stop_id = s.stop_id
            WHERE st.trip_id = ?
            ORDER BY CAST(st.stop_sequence AS INT) ASC;
        """
        cursor.execute(query_stops, (best_trip_id,))
        line_stops = cursor.fetchall()
        conn.close()

        stop_ids_ordered = [r[1] for r in line_stops]
        stop_names_dict = {r[1]: r[2] for r in line_stops}

        if orig_id in stop_ids_ordered and dest_id in stop_ids_ordered:
            idx1 = stop_ids_ordered.index(orig_id)
            idx2 = stop_ids_ordered.index(dest_id)
            
            if idx1 <= idx2:
                path_ids = stop_ids_ordered[idx1:idx2+1]
            else:
                path_ids = stop_ids_ordered[idx2:idx1+1][::-1]

            path_names = [stop_names_dict[sid] for sid in path_ids]
            num_stops = len(path_names) - 1
            est_minutes = num_stops * 2  # Approx 2 mins per station hop

            fare_info = self.get_fare(orig_name, dest_name)
            price = fare_info.get("price", "N/A")

            return {
                "origin_station": orig_name,
                "destination_station": dest_name,
                "num_stops": num_stops,
                "estimated_minutes": est_minutes,
                "fare_price": price,
                "intermediate_stations": path_names,
                "formatted": f"Route from {orig_name} to {dest_name}: {num_stops} stations ({est_minutes} mins). Ticket Price: Rs. {price}. Stations: {' -> '.join(path_names)}"
            }
        else:
            # Fallback NetworkX graph shortest path across all GTFS trip edges
            import networkx as nx
            G = nx.Graph()
            conn2 = self._get_connection()
            c2 = conn2.cursor()
            c2.execute("""
                SELECT DISTINCT st1.stop_id, st2.stop_id, s1.stop_name, s2.stop_name
                FROM stop_times st1
                JOIN stop_times st2 ON st1.trip_id = st2.trip_id AND CAST(st2.stop_sequence AS INT) = CAST(st1.stop_sequence AS INT) + 1
                JOIN stops s1 ON st1.stop_id = s1.stop_id
                JOIN stops s2 ON st2.stop_id = s2.stop_id;
            """)
            edges = c2.fetchall()
            conn2.close()
            name_map = {}
            for u, v, uname, vname in edges:
                G.add_edge(u, v)
                name_map[u] = uname
                name_map[v] = vname

            if orig_id in G and dest_id in G and nx.has_path(G, orig_id, dest_id):
                path_ids = nx.shortest_path(G, orig_id, dest_id)
                path_names = [name_map.get(sid, sid) for sid in path_ids]
                num_stops = len(path_names) - 1
                est_minutes = num_stops * 2
                fare_info = self.get_fare(orig_name, dest_name)
                price = fare_info.get("price", "N/A")
                return {
                    "origin_station": orig_name,
                    "destination_station": dest_name,
                    "num_stops": num_stops,
                    "estimated_minutes": est_minutes,
                    "fare_price": price,
                    "intermediate_stations": path_names,
                    "formatted": f"Route from {orig_name} to {dest_name}: {num_stops} stations ({est_minutes} mins). Ticket Price: Rs. {price}. Stations: {' -> '.join(path_names)}"
                }
            return {"error": f"Route path not found between {orig_name} and {dest_name}."}

    def get_next_trains(self, station: str, current_time_str: str = None):
        conn = self._get_connection()
        sid, sname = self._fuzzy_match_station(conn, station)
        if not sid:
            conn.close()
            return {"error": f"Station '{station}' not found."}

        if not current_time_str:
            current_time_str = datetime.now().strftime("%H:%M:%S")

        cursor = conn.cursor()
        query = """
            SELECT st.arrival_time, st.departure_time, t.direction_id
            FROM stop_times st
            JOIN trips t ON st.trip_id = t.trip_id
            WHERE st.stop_id = ? AND st.departure_time >= ?
            ORDER BY st.departure_time ASC
            LIMIT 5;
        """
        cursor.execute(query, (sid, current_time_str))
        rows = cursor.fetchall()
        conn.close()

        trains = []
        for arr, dep, dir_id in rows:
            destination_headsign = "Aluva" if str(dir_id) == "1" else "Tripunithura"
            trains.append({"arrival_time": arr, "departure_time": dep, "destination": destination_headsign})

        return {
            "station": sname,
            "query_time": current_time_str,
            "upcoming_trains": trains,
            "formatted": f"Next trains at {sname} after {current_time_str}: " + ", ".join([f"{t['departure_time']} (towards {t['destination']})" for t in trains])
        }

if __name__ == "__main__":
    tool = GTFSTool()
    print(tool.get_fare("Aluva", "Pettah"))
    print(tool.find_route("Edapally", "MG Road"))
    print(tool.get_next_trains("Kalamassery", "09:00:00"))
