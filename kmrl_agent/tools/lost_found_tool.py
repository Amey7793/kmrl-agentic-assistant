"""
===============================================================================
TOOL MODULE: Lost & Found Query Engine (lost_found_tool.py)
PURPOSE:
  Executes high-speed SQLite FTS5 (Full-Text Search) and structured SQL queries
  over 31,626 lost property records in 'lost_found.db'.

KEY FUNCTIONS:
  1. search_lost_items(query, category, station, limit): FTS keyword & field search.
  2. get_item_by_lfid(lfid): Exact lookup by Lost & Found Tracking ID (e.g. ALVA20260809-001).
===============================================================================
"""

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "lost_found.db")

class LostFoundTool:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def search_lost_items(self, query: str = None, category: str = None, station: str = None, limit: int = 10):
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if query:
            # FTS search or direct match
            conditions.append("(rowid IN (SELECT rowid FROM lost_items_fts WHERE lost_items_fts MATCH ?) OR category LIKE ? OR lfid LIKE ?)")
            fts_query = f'"{query.strip()}"*'
            like_query = f"%{query.strip()}%"
            params.extend([fts_query, like_query, like_query])

        if category:
            conditions.append("category LIKE ?")
            params.append(f"%{category.strip()}%")

        if station:
            conditions.append("(found_station LIKE ? OR current_station LIKE ?)")
            params.extend([f"%{station.strip()}%", f"%{station.strip()}%"])

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        sql = f"""
            SELECT sl_no, found_date, category, found_station, current_station, lfid
            FROM lost_items
            {where_clause}
            ORDER BY found_date DESC
            LIMIT ?;
        """
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for r in rows:
            results.append({
                "sl_no": r[0],
                "found_date": r[1],
                "category": r[2],
                "found_station": r[3],
                "current_station": r[4],
                "lfid": r[5]
            })

        if results:
            formatted_summary = f"Found {len(results)} matching lost item(s):\n" + "\n".join([
                f"- [LFID: {item['lfid']}] {item['category']} found on {item['found_date']} at {item['found_station']} station (Currently stored at: {item['current_station']})."
                for item in results
            ])
        else:
            formatted_summary = "No matching lost items found in the database for your query criteria."

        return {
            "count": len(results),
            "items": results,
            "formatted": formatted_summary
        }

    def get_item_by_lfid(self, lfid: str):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sl_no, found_date, category, found_station, current_station, lfid
            FROM lost_items WHERE lfid = ?;
        """, (lfid.strip(),))
        r = cursor.fetchone()
        conn.close()

        if r:
            item = {
                "sl_no": r[0],
                "found_date": r[1],
                "category": r[2],
                "found_station": r[3],
                "current_station": r[4],
                "lfid": r[5]
            }
            return {
                "item": item,
                "formatted": f"Item Found! LFID: {item['lfid']} | Category: {item['category']} | Date Found: {item['found_date']} | Found At: {item['found_station']} | Currently At: {item['current_station']}"
            }
        else:
            return {"error": f"No item found with LFID '{lfid}'."}

if __name__ == "__main__":
    tool = LostFoundTool()
    print(tool.get_item_by_lfid("ALVA20260809-001"))
    print(tool.search_lost_items(query="UMBRELLA", station="Aluva", limit=3))
