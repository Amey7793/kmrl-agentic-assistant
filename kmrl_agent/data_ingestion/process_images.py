"""
===============================================================================
MODULE: Visual Image Processing Script (process_images.py)
PURPOSE:
  Extracts text content from visual infographics, maps, parking charts, and feeder
  bus schedule images in 'Knowledge Base/Images/' using RapidOCR into structured
  Markdown tables and metadata.
  Saves the processed visual records to 'processed_images.json'.
===============================================================================
"""

import os
import json
import re
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(os.path.dirname(BASE_DIR), "Knowledge Base", "Images")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_images.json")

# Initialize RapidOCR engine
ocr_engine = RapidOCR()


def extract_ocr_text(img_path):
    """
    Extracts ordered text lines and spatial bounding-box items from an image using RapidOCR.
    Returns (lines, ocr_items).
    """
    try:
        results, _ = ocr_engine(img_path)
        if not results:
            return [], []
        
        lines = []
        ocr_items = []
        for item in results:
            text = item[1].strip() if item[1] else ""
            score = float(item[2]) if item[2] else 0.0
            if text and score > 0.3:
                lines.append(text)
                box = item[0]
                pts = np.array(box)
                x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                h, w = y2 - y1, x2 - x1
                ocr_items.append({
                    "box": box,
                    "text": text,
                    "score": score,
                    "bbox": [x1, y1, x2, y2],
                    "cx": cx,
                    "cy": cy,
                    "w": w,
                    "h": h
                })
        return lines, ocr_items
    except Exception as e:
        print(f"  [OCR Warning] Failed to run OCR on {os.path.basename(img_path)}: {e}", flush=True)
        return [], []


def infer_metadata(filename, lines):
    """Rule-based title and category detection from filename and OCR text lines."""
    name_clean = os.path.splitext(filename)[0].replace("-", " ").replace("_", " ").title()
    full_text = " ".join(lines).lower()
    lower_name = filename.lower()

    if "parking" in lower_name or "parking" in full_text:
        category = "Parking Rates"
        title = "Kochi Metro Station Parking Fee Chart"
    elif "feeder" in lower_name or "bus" in full_text:
        category = "Feeder Bus Timings"
        title = f"Feeder Bus Schedule - {name_clean}"
    elif "fare" in lower_name or "tariff" in full_text or "slab" in full_text:
        category = "Fare Chart Infographic"
        title = "Kochi Metro Fare Chart & Tariff Structure"
    elif "map" in lower_name or "system" in full_text:
        category = "System Map"
        title = "Kochi Metro Network & Water Metro System Map"
    else:
        category = "Infographic OCR"
        title = f"KMRL Visual Asset - {name_clean}"

    return title, category


def reconstruct_markdown_table(ocr_items, img_width):
    """
    Reconstructs rows and columns from OCR bounding-box coordinates.
    Returns (table_md_string, top_lines, bottom_lines) or None if no table detected.
    """
    if not ocr_items:
        return None

    sorted_items = sorted(ocr_items, key=lambda it: (it["cy"], it["cx"]))
    heights = [it["h"] for it in sorted_items]
    median_h = float(np.median(heights)) if heights else 20.0

    # Group items into physical horizontal lines based on cy
    y_tol = max(15.0, median_h * 0.65)
    phys_lines = []
    for it in sorted_items:
        placed = False
        for pline in phys_lines:
            line_cy = float(np.mean([item["cy"] for item in pline]))
            if abs(it["cy"] - line_cy) < y_tol:
                pline.append(it)
                placed = True
                break
        if not placed:
            phys_lines.append([it])

    for pline in phys_lines:
        pline.sort(key=lambda it: it["cx"])

    grid_lines = [pline for pline in phys_lines if len(pline) >= 2]
    if len(grid_lines) < 2:
        return None

    # Cluster X coordinates across multi-item lines to find columns
    all_cx = sorted([it["cx"] for pline in grid_lines for it in pline])
    x_tol = max(45.0, img_width * 0.04) if img_width > 0 else 50.0

    col_clusters = []
    for cx in all_cx:
        if not col_clusters:
            col_clusters.append([cx])
        else:
            if abs(cx - np.mean(col_clusters[-1])) < x_tol:
                col_clusters[-1].append(cx)
            else:
                col_clusters.append([cx])

    col_centers = [float(np.mean(c)) for c in col_clusters]
    num_cols = len(col_centers)

    if num_cols < 2:
        return None

    def get_col_idx(item):
        best_c = int(np.argmin([abs(item["cx"] - c) for c in col_centers]))
        if abs(item["cx"] - col_centers[best_c]) < x_tol * 1.5:
            return best_c
        return None

    # Group lines into top standalone text, table lines, bottom standalone text
    top_text = []
    table_lines = []
    bottom_text = []
    in_table = False

    for pline in phys_lines:
        cols_hit = set(get_col_idx(it) for it in pline)
        cols_hit.discard(None)
        
        if in_table and table_lines:
            prev_line_cy = float(np.mean([it["cy"] for it in table_lines[-1]]))
            curr_line_cy = float(np.mean([it["cy"] for it in pline]))
            if curr_line_cy - prev_line_cy > median_h * 3.5:
                in_table = False

        is_table_line = (len(cols_hit) >= 2 or (in_table and len(cols_hit) >= 1))
        
        if is_table_line:
            table_lines.append(pline)
            in_table = True
        else:
            if not table_lines:
                top_text.append(pline)
            else:
                bottom_text.append(pline)

    if len(table_lines) < 2:
        return None

    # Group physical table lines into logical table rows
    logical_rows = []
    curr_row_items = []
    
    for pline in table_lines:
        has_col0 = any(get_col_idx(it) == 0 for it in pline)
        if curr_row_items:
            prev_cy = float(np.mean([it["cy"] for it in curr_row_items]))
            curr_cy = float(np.mean([it["cy"] for it in pline]))
            if has_col0 and (curr_cy - prev_cy > median_h * 0.8):
                logical_rows.append(curr_row_items)
                curr_row_items = list(pline)
            elif curr_cy - prev_cy > median_h * 2.0:
                logical_rows.append(curr_row_items)
                curr_row_items = list(pline)
            else:
                curr_row_items.extend(pline)
        else:
            curr_row_items.extend(pline)
            
    if curr_row_items:
        logical_rows.append(curr_row_items)

    # Build 2D matrix of row cells
    table_matrix = []
    for r_items in logical_rows:
        row_cells = [[] for _ in range(num_cols)]
        r_items_sorted = sorted(r_items, key=lambda it: (it["cy"], it["cx"]))
        for it in r_items_sorted:
            c_idx = get_col_idx(it)
            if c_idx is not None:
                row_cells[c_idx].append(it["text"])
        
        row_str_cells = []
        for cell_texts in row_cells:
            joined = " ".join(cell_texts)
            if "Four wheeler" in joined and "Three wheeler" in joined:
                joined = joined.replace("Four wheeler", "Four/Three Wheeler").replace("Three wheeler", "").strip()
                joined = " ".join(joined.split())
            row_str_cells.append(joined)
        table_matrix.append(row_str_cells)

    if not table_matrix:
        return None

    header = table_matrix[0]
    if not header[0]:
        header[0] = "Parking Type"

    md_lines = []
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    for drow in table_matrix[1:]:
        md_lines.append("| " + " | ".join(drow) + " |")

    top_str_lines = [" ".join(it["text"] for it in line) for line in top_text]
    bottom_str_lines = [" ".join(it["text"] for it in line) for line in bottom_text]

    return "\n".join(md_lines), top_str_lines, bottom_str_lines


def format_ocr_to_markdown(title, category, filename, lines, ocr_items=None, img_width=0):
    """Formats OCR results into Markdown, reconstructing tables when spatial coordinates form a grid."""
    if not lines:
        return f"### {title}\n*No readable text detected in visual asset `{filename}`.*"

    table_res = reconstruct_markdown_table(ocr_items, img_width) if ocr_items else None

    if table_res:
        table_md, top_lines, bottom_lines = table_res
        sections = [f"### {title}", f"**Category**: {category}", f"**Source File**: `{filename}`\n", "**Extracted Content (OCR)**:"]
        if top_lines:
            sections.extend(top_lines)
            sections.append("")
        sections.append(table_md)
        if bottom_lines:
            sections.append("")
            sections.extend(bottom_lines)
        return "\n".join(sections)
    else:
        bullets = "\n".join([f"- {line}" for line in lines])
        return f"### {title}\n**Category**: {category}\n**Source File**: `{filename}`\n\n**Extracted Content (OCR)**:\n{bullets}"


def process_images():
    print(f"Processing image assets in: {IMAGES_DIR} using RapidOCR...", flush=True)
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: {IMAGES_DIR} not found!", flush=True)
        return

    processed_records = []
    image_files = sorted(os.listdir(IMAGES_DIR))

    for idx, img_name in enumerate(image_files, 1):
        img_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.isfile(img_path):
            continue

        print(f"[{idx}/{len(image_files)}] Extracting OCR from: {img_name}", flush=True)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                format_name = img.format
        except Exception:
            width, height, format_name = 0, 0, "UNKNOWN"

        lines, ocr_items = extract_ocr_text(img_path)
        print(f"   -> Extracted {len(lines)} lines of text.", flush=True)

        title, category = infer_metadata(img_name, lines)
        content_markdown = format_ocr_to_markdown(title, category, img_name, lines, ocr_items, width)

        processed_records.append({
            "filename": img_name,
            "filepath": img_path,
            "title": title,
            "category": category,
            "content_markdown": content_markdown,
            "raw_ocr": lines,
            "width": width,
            "height": height
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully processed {len(processed_records)} image assets into {OUTPUT_PATH}.", flush=True)


if __name__ == "__main__":
    process_images()

