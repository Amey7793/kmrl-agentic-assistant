"""
===============================================================================
TOOL MODULE: Visual Infographic Data Engine (vision_tool.py)
PURPOSE:
  Queries pre-processed markdown tables and structured metadata derived from 
  visual infographic assets ('processed_images.json').

KEY ASSETS COVERED:
  1. Parking-Rate.jpeg: Two-wheeler, car, auto, and overnight parking fee schedules.
  2. SYSTEM-MAP: Metro Line 1 & Water Metro network layout & interchanges.
  3. Feeder Bus Timetables: Kalamassery, Tripunithura, and feeder electric bus schedules.
  4. Fare Chart Infographic: Official KMRL fare slabs, rules, and discounts.
===============================================================================
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_IMAGES_PATH = os.path.join(BASE_DIR, "processed_images.json")

class VisionTool:
    def __init__(self, json_path=PROCESSED_IMAGES_PATH):
        self.json_path = json_path
        self.records = self._load_data()

    def _load_data(self):
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def query_visual_data(self, query: str):
        query_clean = query.lower().strip()
        stop_words = {"what", "are", "the", "for", "and", "is", "in", "at", "of", "to", "a", "show", "tell", "me"}
        query_words = [w for w in query_clean.split() if w not in stop_words]

        scored_records = []
        for rec in self.records:
            title = rec.get("title", "").lower()
            category = rec.get("category", "").lower()
            content = rec.get("content_markdown", "").lower()
            text_block = f"{title} {category} {content}"

            score = sum(1 for w in query_words if w in text_block)
            if score > 0:
                scored_records.append((score, rec))

        scored_records.sort(key=lambda x: x[0], reverse=True)
        matched = [r[1] for r in scored_records]

        if not matched:
            matched = self.records  # fallback to all infographics if no keyword match

        formatted_res = "\n\n---\n\n".join([
            f"### Visual Asset: {r['title']} ({r['filename']})\nCategory: {r['category']}\n\n{r['content_markdown']}"
            for r in matched[:3]
        ])

        return {
            "matched_count": len(matched),
            "records": matched[:3],
            "formatted": formatted_res
        }

if __name__ == "__main__":
    tool = VisionTool()
    print(tool.query_visual_data("parking charges for bikes and cars"))
