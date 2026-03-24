"""
LocalROSA – Report Generator
==============================
Generates PDF reports and CSV exports for ROSA assessments.
Uses FPDF2 for PDF generation (pure Python, no external dependencies).
"""

import csv
import io
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def generate_pdf_report(
    original_image_path: str,
    annotated_image_path: str,
    rosa_result,
    pose_angles=None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a detailed PDF report for a single ROSA assessment.

    Args:
        original_image_path: Path to the original uploaded image.
        annotated_image_path: Path to the annotated image with landmarks.
        rosa_result: ROSAResult dataclass.
        pose_angles: Optional PoseAngles dataclass.
        output_path: Optional output path. If None, saves to results/.

    Returns:
        Path to the generated PDF file.
    """
    from fpdf import FPDF

    if output_path is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/rosa_report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Summary ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "LocalROSA - Ergonomic Assessment Report", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # Grand Score Box
    pdf.set_font("Helvetica", "B", 16)
    score = rosa_result.grand_rosa_score
    risk = rosa_result.risk_level

    # Color based on risk
    if risk == "Low":
        pdf.set_fill_color(40, 167, 69)
    elif risk == "Moderate":
        pdf.set_fill_color(255, 193, 7)
    else:
        pdf.set_fill_color(220, 53, 69)

    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, f"Grand ROSA Score: {score}/10 - Risk Level: {risk}", new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Images side by side
    try:
        img_w = 90
        if os.path.exists(original_image_path):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(95, 8, "Original Image", align="C")
            pdf.cell(95, 8, "Annotated Analysis", new_x="LMARGIN", new_y="NEXT", align="C")
            y_pos = pdf.get_y()
            pdf.image(original_image_path, x=10, y=y_pos, w=img_w)
        if os.path.exists(annotated_image_path):
            pdf.image(annotated_image_path, x=105, y=y_pos, w=img_w)
        pdf.ln(75)
    except Exception as e:
        logger.warning(f"Could not add images to PDF: {e}")
        pdf.ln(5)

    # ── Page 2: Detailed Breakdown ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Detailed ROSA Score Breakdown", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Section A: Chair
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 8, "Section A: Chair Assessment", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 10)

    chair_items = [
        ("Seat Height + Depth Score", rosa_result.seat_height_depth_score),
        ("Armrest + Back Support Score", rosa_result.armrest_back_score),
        ("Chair Score (raw)", rosa_result.chair_score_raw),
        ("Chair Score (with duration)", rosa_result.chair_score_final),
    ]
    for label, val in chair_items:
        pdf.cell(120, 7, f"  {label}", border="B")
        pdf.cell(60, 7, str(val), border="B", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(3)

    # Section B: Monitor & Phone
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Section B: Monitor & Telephone", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 10)

    mp_items = [
        ("Monitor Score", rosa_result.monitor_score_raw),
        ("Phone Score", rosa_result.phone_score_raw),
        ("Monitor+Phone Combined (raw)", rosa_result.monitor_phone_score_raw),
        ("Monitor+Phone (with duration)", rosa_result.monitor_phone_score_final),
    ]
    for label, val in mp_items:
        pdf.cell(120, 7, f"  {label}", border="B")
        pdf.cell(60, 7, str(val), border="B", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(3)

    # Section C: Keyboard & Mouse
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Section C: Keyboard & Mouse", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 10)

    km_items = [
        ("Keyboard Score", rosa_result.keyboard_score_raw),
        ("Mouse Score", rosa_result.mouse_score_raw),
        ("Keyboard+Mouse Combined (raw)", rosa_result.keyboard_mouse_score_raw),
        ("Keyboard+Mouse (with duration)", rosa_result.keyboard_mouse_score_final),
    ]
    for label, val in km_items:
        pdf.cell(120, 7, f"  {label}", border="B")
        pdf.cell(60, 7, str(val), border="B", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(3)

    # Final Scores
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Final Scores", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_font("Helvetica", "", 10)

    final_items = [
        ("Peripherals/Monitor Score", rosa_result.peripherals_monitor_score),
        ("Grand ROSA Score", rosa_result.grand_rosa_score),
        ("Risk Level", rosa_result.risk_level),
    ]
    for label, val in final_items:
        pdf.cell(120, 7, f"  {label}", border="B")
        pdf.cell(60, 7, str(val), border="B", new_x="LMARGIN", new_y="NEXT", align="C")

    # Detected Angles
    if pose_angles:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Detected Angles", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 10)

        angles = [
            ("Knee Angle", f"{pose_angles.knee_angle:.1f} deg"),
            ("Elbow Angle", f"{pose_angles.elbow_angle:.1f} deg"),
            ("Trunk Recline", f"{pose_angles.trunk_recline_angle:.1f} deg"),
            ("Neck Flexion", f"{pose_angles.neck_flexion:.1f} deg"),
            ("Wrist Extension", f"{pose_angles.wrist_extension:.1f} deg"),
            ("Shoulder Shrug", "Yes" if pose_angles.shoulder_shrug else "No"),
            ("Feet on Floor", "Yes" if pose_angles.feet_on_floor else "No"),
            ("Detection Confidence", f"{pose_angles.detection_confidence:.0%}"),
            ("Side View", f"{'Yes' if pose_angles.is_valid_side_view else 'No'} ({pose_angles.side_view_confidence:.0%})"),
        ]
        for label, val in angles:
            pdf.cell(120, 7, f"  {label}", border="B")
            pdf.cell(60, 7, val, border="B", new_x="LMARGIN", new_y="NEXT", align="C")

    # ── Page 3: Recommendations ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Ergonomic Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    for i, rec in enumerate(rosa_result.recommendations, 1):
        pdf.multi_cell(0, 7, f"{i}. {rec}")
        pdf.ln(2)

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Generated by LocalROSA - AI-Powered Rapid Office Strain Assessment Engine", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 5, "Based on ROSA methodology by Sonne, Villalta & Andrews (2012)", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.output(output_path)
    logger.info(f"PDF report saved: {output_path}")
    return output_path


def generate_csv_export(
    results: List[Dict],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a CSV export for batch results.

    Args:
        results: List of dicts with filename, scores, etc.
        output_path: Optional output path.

    Returns:
        Path to the CSV file.
    """
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/rosa_batch_{timestamp}.csv"

    fieldnames = [
        "filename",
        "grand_rosa_score",
        "risk_level",
        "chair_score",
        "monitor_phone_score",
        "keyboard_mouse_score",
        "peripherals_monitor_score",
        "knee_angle",
        "elbow_angle",
        "trunk_recline",
        "neck_flexion",
        "wrist_extension",
        "detection_confidence",
        "is_side_view",
        "warnings",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logger.info(f"CSV export saved: {output_path}")
    return output_path


def generate_batch_summary(results: List[Dict]) -> Dict:
    """
    Generate summary statistics for a batch of ROSA assessments.

    Returns:
        Dictionary with summary stats.
    """
    if not results:
        return {"count": 0}

    scores = [r.get("grand_rosa_score", 0) for r in results]
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0}
    for r in results:
        level = r.get("risk_level", "")
        if level in risk_counts:
            risk_counts[level] += 1

    return {
        "count": len(results),
        "average_score": round(sum(scores) / len(scores), 1),
        "min_score": min(scores),
        "max_score": max(scores),
        "highest_risk_image": max(results, key=lambda x: x.get("grand_rosa_score", 0)).get("filename", ""),
        "risk_distribution": risk_counts,
    }
