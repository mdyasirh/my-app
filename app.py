"""
LocalROSA – AI-Powered Rapid Office Strain Assessment (ROSA) Engine
====================================================================
Main application entry point with Gradio web interface.

Run with: python app.py
Access at: http://127.0.0.1:7860

This application processes side-view photos of seated workers and
computes the official ROSA score (1-10) using computer vision.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from pose_detector import PoseDetector, PoseAngles
from rosa_calculator import calculate_rosa_from_angles, ROSAResult
from report_generator import generate_pdf_report, generate_csv_export, generate_batch_summary
from utils import (
    setup_logging,
    load_image,
    save_annotated_image,
    extract_zip,
    get_risk_color_hex,
    validate_image_file,
    ensure_results_dir,
    SUPPORTED_FORMATS,
)

setup_logging()
logger = logging.getLogger(__name__)

# Global pose detector instance
detector = PoseDetector(min_detection_confidence=0.5)


# ─────────────────────────────────────────────────────────────
# Core Processing Functions
# ─────────────────────────────────────────────────────────────

def process_single_image(
    image_path: str,
    hours_per_day: float = 8.0,
    armrests_present: bool = True,
    lumbar_support: bool = True,
    non_adjustable_chair: bool = False,
    non_adjustable_armrest: bool = False,
    non_adjustable_back: bool = False,
    non_adjustable_keyboard: bool = False,
    screen_glare: bool = False,
    no_document_holder: bool = False,
    phone_use: str = "None",
    # Manual overrides
    override_chair: Optional[int] = None,
    override_monitor_phone: Optional[int] = None,
    override_keyboard_mouse: Optional[int] = None,
    override_justification: str = "",
) -> Tuple[Optional[np.ndarray], Optional[ROSAResult], Optional[PoseAngles], str]:
    """
    Process a single image through the full ROSA pipeline.

    Returns:
        (annotated_image, rosa_result, pose_angles, status_message)
    """
    # Load and validate image
    image = load_image(image_path)
    if image is None:
        return None, None, None, "Failed to load image. Please check the file format."

    # Detect pose
    pose_angles = detector.detect_pose(image)
    if pose_angles is None:
        return None, None, None, (
            "Could not detect a person in the image. Tips:\n"
            "- Ensure the full body (head to feet) is visible\n"
            "- Use a clear side-view angle\n"
            "- Ensure good lighting"
        )

    # Map phone use string to code
    phone_map = {"None": "none", "Hands-free": "hands_free", "In hand": "reaching", "Neck/Shoulder hold": "neck_shoulder"}
    phone_code = phone_map.get(phone_use, "none")

    # Calculate ROSA score from detected angles
    rosa_result = calculate_rosa_from_angles(
        knee_angle=pose_angles.knee_angle,
        elbow_angle=pose_angles.elbow_angle,
        trunk_recline_angle=pose_angles.trunk_recline_angle,
        neck_flexion=pose_angles.neck_flexion,
        wrist_extension=pose_angles.wrist_extension,
        shoulder_shrug=pose_angles.shoulder_shrug,
        feet_on_floor=pose_angles.feet_on_floor,
        space_behind_knee_ok=pose_angles.space_behind_knee_ok,
        armrests_present=armrests_present,
        lumbar_support=lumbar_support,
        monitor_distance_ok=True,
        mouse_in_line=True,
        mouse_reaching=False,
        phone_use=phone_code,
        hours_per_day=hours_per_day,
        non_adjustable_seat=non_adjustable_chair,
        non_adjustable_armrest=non_adjustable_armrest,
        non_adjustable_back=non_adjustable_back,
        non_adjustable_keyboard=non_adjustable_keyboard,
        screen_glare=screen_glare,
        no_document_holder=no_document_holder,
    )

    # Apply manual overrides if provided
    if override_chair is not None and override_chair > 0:
        rosa_result.chair_score_final = override_chair
        rosa_result.breakdown["Chair Score (with duration)"] = f"{override_chair} [MANUAL OVERRIDE]"
    if override_monitor_phone is not None and override_monitor_phone > 0:
        rosa_result.monitor_phone_score_final = override_monitor_phone
        rosa_result.breakdown["Monitor+Phone (with duration)"] = f"{override_monitor_phone} [MANUAL OVERRIDE]"
    if override_keyboard_mouse is not None and override_keyboard_mouse > 0:
        rosa_result.keyboard_mouse_score_final = override_keyboard_mouse
        rosa_result.breakdown["Keyboard+Mouse (with duration)"] = f"{override_keyboard_mouse} [MANUAL OVERRIDE]"

    if override_justification:
        rosa_result.breakdown["Override Justification"] = override_justification

    # Draw annotated image
    annotated = detector.draw_annotated_image(image, pose_angles)

    # Build status message
    warnings_text = ""
    if pose_angles.warnings:
        warnings_text = "\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in pose_angles.warnings)

    status = (
        f"Detection successful! Confidence: {pose_angles.detection_confidence:.0%}\n"
        f"Side: {pose_angles.detected_side} | Side-view: {'Yes' if pose_angles.is_valid_side_view else 'No'}"
        f"{warnings_text}"
    )

    return annotated, rosa_result, pose_angles, status


# ─────────────────────────────────────────────────────────────
# Gradio UI Callback Functions
# ─────────────────────────────────────────────────────────────

def analyze_single(
    image,
    hours_per_day,
    armrests_present,
    lumbar_support,
    non_adj_chair,
    non_adj_armrest,
    non_adj_back,
    non_adj_keyboard,
    screen_glare,
    no_doc_holder,
    phone_use,
    override_chair,
    override_mp,
    override_km,
    override_justification,
):
    """Gradio callback for single image analysis."""
    if image is None:
        return None, "Please upload an image.", "", "", gr.update(visible=False)

    # Save uploaded image to temp file
    temp_path = os.path.join(tempfile.gettempdir(), "localrosa_upload.png")
    if isinstance(image, np.ndarray):
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif isinstance(image, str):
        temp_path = image

    annotated, result, angles, status = process_single_image(
        temp_path,
        hours_per_day=hours_per_day,
        armrests_present=armrests_present,
        lumbar_support=lumbar_support,
        non_adjustable_chair=non_adj_chair,
        non_adjustable_armrest=non_adj_armrest,
        non_adjustable_back=non_adj_back,
        non_adjustable_keyboard=non_adj_keyboard,
        screen_glare=screen_glare,
        no_document_holder=no_doc_holder,
        phone_use=phone_use,
        override_chair=override_chair if override_chair and override_chair > 0 else None,
        override_monitor_phone=override_mp if override_mp and override_mp > 0 else None,
        override_keyboard_mouse=override_km if override_km and override_km > 0 else None,
        override_justification=override_justification or "",
    )

    if result is None:
        return None, status, "", "", gr.update(visible=False)

    # Convert annotated image to RGB for Gradio display
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # Format breakdown table
    breakdown_md = format_breakdown_markdown(result)

    # Format recommendations
    recs_md = "\n".join(f"- {r}" for r in result.recommendations)

    # Score display with color
    color = get_risk_color_hex(result.risk_level)
    score_html = f"""
    <div style="text-align:center; padding:20px; border-radius:10px; background:{color}20; border:2px solid {color};">
        <h1 style="color:{color}; margin:0; font-size:3em;">{result.grand_rosa_score}</h1>
        <h3 style="color:{color}; margin:5px 0;">/ 10</h3>
        <h2 style="margin:5px 0;">Risk Level: <span style="color:{color};">{result.risk_level}</span></h2>
    </div>
    """

    # Save results
    try:
        ensure_results_dir()
        ann_path = save_annotated_image(
            annotated, os.path.basename(temp_path)
        )
        # Save breakdown to JSON
        json_path = ann_path.replace(".png", ".json")
        with open(json_path, "w") as f:
            json.dump(result.breakdown, f, indent=2)
        status += f"\n\nResults saved to: {ann_path}"
    except Exception as e:
        logger.error(f"Error saving results: {e}")

    return annotated_rgb, score_html, breakdown_md, recs_md, gr.update(visible=True)


def analyze_batch(files, hours_per_day, armrests_present, lumbar_support, progress=gr.Progress()):
    """Gradio callback for batch image analysis."""
    if not files:
        return "No files uploaded.", None, ""

    all_results = []
    summary_lines = []
    annotated_images = []

    progress(0, desc="Starting batch analysis...")

    for i, file_obj in enumerate(files):
        file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        filename = os.path.basename(file_path)

        progress((i + 1) / len(files), desc=f"Processing {filename}...")

        # Handle ZIP files
        if file_path.lower().endswith(".zip"):
            extracted = extract_zip(file_path)
            for ext_path in extracted:
                result_dict = _process_batch_item(ext_path, hours_per_day, armrests_present, lumbar_support)
                if result_dict:
                    all_results.append(result_dict)
            continue

        result_dict = _process_batch_item(file_path, hours_per_day, armrests_present, lumbar_support)
        if result_dict:
            all_results.append(result_dict)

    if not all_results:
        return "No images could be processed. Please check the files and try again.", None, ""

    # Generate summary
    summary = generate_batch_summary(all_results)

    # Generate CSV
    csv_path = generate_csv_export(all_results)

    # Format summary
    summary_md = f"""## Batch Analysis Complete

| Metric | Value |
|--------|-------|
| Images Processed | {summary['count']} |
| Average ROSA Score | {summary['average_score']} |
| Minimum Score | {summary['min_score']} |
| Maximum Score | {summary['max_score']} |
| Highest Risk Image | {summary['highest_risk_image']} |

### Risk Distribution
| Level | Count |
|-------|-------|
| Low (1-3) | {summary['risk_distribution']['Low']} |
| Moderate (4-5) | {summary['risk_distribution']['Moderate']} |
| High (6-10) | {summary['risk_distribution']['High']} |

### Individual Results
"""
    for r in all_results:
        color = get_risk_color_hex(r.get("risk_level", ""))
        summary_md += f"- **{r['filename']}**: Score **{r['grand_rosa_score']}** ({r['risk_level']})\n"

    return summary_md, csv_path, f"CSV saved to: {csv_path}"


def _process_batch_item(file_path, hours_per_day, armrests_present, lumbar_support):
    """Process a single item in batch mode."""
    filename = os.path.basename(file_path)

    annotated, result, angles, status = process_single_image(
        file_path,
        hours_per_day=hours_per_day,
        armrests_present=armrests_present,
        lumbar_support=lumbar_support,
    )

    if result is None:
        return None

    # Save annotated image
    if annotated is not None:
        save_annotated_image(annotated, filename)

    return {
        "filename": filename,
        "grand_rosa_score": result.grand_rosa_score,
        "risk_level": result.risk_level,
        "chair_score": result.chair_score_final,
        "monitor_phone_score": result.monitor_phone_score_final,
        "keyboard_mouse_score": result.keyboard_mouse_score_final,
        "peripherals_monitor_score": result.peripherals_monitor_score,
        "knee_angle": round(angles.knee_angle, 1) if angles else 0,
        "elbow_angle": round(angles.elbow_angle, 1) if angles else 0,
        "trunk_recline": round(angles.trunk_recline_angle, 1) if angles else 0,
        "neck_flexion": round(angles.neck_flexion, 1) if angles else 0,
        "wrist_extension": round(angles.wrist_extension, 1) if angles else 0,
        "detection_confidence": round(angles.detection_confidence, 2) if angles else 0,
        "is_side_view": angles.is_valid_side_view if angles else False,
        "warnings": "; ".join(angles.warnings) if angles else "",
    }


def export_pdf(image, hours_per_day, armrests_present, lumbar_support,
               non_adj_chair, non_adj_armrest, non_adj_back, non_adj_keyboard,
               screen_glare, no_doc_holder, phone_use,
               override_chair, override_mp, override_km, override_justification):
    """Generate and return PDF report for download."""
    if image is None:
        return None

    temp_path = os.path.join(tempfile.gettempdir(), "localrosa_upload.png")
    if isinstance(image, np.ndarray):
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    elif isinstance(image, str):
        temp_path = image

    annotated, result, angles, status = process_single_image(
        temp_path,
        hours_per_day=hours_per_day,
        armrests_present=armrests_present,
        lumbar_support=lumbar_support,
        non_adjustable_chair=non_adj_chair,
        non_adjustable_armrest=non_adj_armrest,
        non_adjustable_back=non_adj_back,
        non_adjustable_keyboard=non_adj_keyboard,
        screen_glare=screen_glare,
        no_document_holder=no_doc_holder,
        phone_use=phone_use,
        override_chair=override_chair if override_chair and override_chair > 0 else None,
        override_monitor_phone=override_mp if override_mp and override_mp > 0 else None,
        override_keyboard_mouse=override_km if override_km and override_km > 0 else None,
        override_justification=override_justification or "",
    )

    if result is None:
        return None

    # Save annotated image temporarily
    ann_path = os.path.join(tempfile.gettempdir(), "localrosa_annotated.png")
    cv2.imwrite(ann_path, annotated)

    pdf_path = generate_pdf_report(temp_path, ann_path, result, angles)
    return pdf_path


def format_breakdown_markdown(result: ROSAResult) -> str:
    """Format ROSA result as a Markdown table."""
    rows = []
    rows.append("| Section | Score |")
    rows.append("|---------|-------|")

    section_map = {
        "Section A: Chair": [
            ("Seat Height + Depth", result.seat_height_depth_score),
            ("Armrest + Back Support", result.armrest_back_score),
            ("Chair Score (raw)", result.chair_score_raw),
            ("Chair Score (final)", result.chair_score_final),
        ],
        "Section B: Monitor & Phone": [
            ("Monitor", result.monitor_score_raw),
            ("Phone", result.phone_score_raw),
            ("Monitor+Phone (raw)", result.monitor_phone_score_raw),
            ("Monitor+Phone (final)", result.monitor_phone_score_final),
        ],
        "Section C: Keyboard & Mouse": [
            ("Keyboard", result.keyboard_score_raw),
            ("Mouse", result.mouse_score_raw),
            ("Keyboard+Mouse (raw)", result.keyboard_mouse_score_raw),
            ("Keyboard+Mouse (final)", result.keyboard_mouse_score_final),
        ],
        "Final": [
            ("Peripherals/Monitor", result.peripherals_monitor_score),
            ("**Grand ROSA Score**", f"**{result.grand_rosa_score}**"),
            ("**Risk Level**", f"**{result.risk_level}**"),
        ],
    }

    for section, items in section_map.items():
        rows.append(f"| **{section}** | |")
        for label, value in items:
            rows.append(f"| {label} | {value} |")

    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────
# Photo Guide Content
# ─────────────────────────────────────────────────────────────

PHOTO_GUIDE = """
## How to Take Perfect Side-View Photos for ROSA Assessment

### Camera Position
- Place the camera **directly to the side** of the seated person (90 degrees from front)
- Camera should be at **waist height** (approximately seat level)
- Distance: **6-10 feet (2-3 meters)** from the subject
- Use landscape orientation

### Subject Requirements
- Full body must be visible: **head to feet**
- The person should be in their **natural working posture**
- Arms should be in their typical working position (on keyboard/mouse)
- Both feet should be visible

### Lighting
- Use **even, bright lighting** - avoid harsh shadows
- Avoid backlighting (don't place bright windows behind the subject)
- Flash is acceptable but natural light is preferred

### What to Avoid
- **Front or back views** (the system needs a side profile)
- Cropped images (missing head, feet, or arms)
- Very dark or overexposed images
- Multiple people in frame
- Obstructed joints (covered by furniture, clothing piles, etc.)

### Best Practices
- Take **2-3 photos** from slightly different angles
- Include the **entire workstation** (chair, desk, monitor) if possible
- Ask the subject to **sit naturally** (not posed)
- Remove obstructions between camera and subject

### Image Specifications
- Minimum resolution: **640x480** (higher is better)
- Supported formats: JPG, PNG, BMP, TIFF, WebP
- Maximum recommended: **4096x4096**
"""


# ─────────────────────────────────────────────────────────────
# Build Gradio Interface
# ─────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Build the complete Gradio application."""

    # Custom CSS for professional styling
    custom_css = """
    .main-title {
        text-align: center;
        padding: 10px;
    }
    .score-display {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
    }
    .risk-low { color: #28a745; }
    .risk-moderate { color: #ffc107; }
    .risk-high { color: #dc3545; }
    .section-header {
        font-weight: bold;
        padding: 8px 0;
        border-bottom: 2px solid #333;
    }
    """

    with gr.Blocks(
        title="LocalROSA - AI-Powered ROSA Assessment",
        css=custom_css,
        theme=gr.themes.Soft(),
    ) as app:

        # Header
        gr.Markdown(
            """
            # LocalROSA - AI-Powered Rapid Office Strain Assessment
            **Compute official ROSA scores (1-10) from side-view photos using computer vision.**
            100% local and offline processing. No data leaves your machine.
            """
        )

        with gr.Tabs() as tabs:

            # ── Tab 1: Single Image Analysis ──
            with gr.TabItem("Single Image Analysis", id="single"):
                with gr.Row():
                    # Left column: Inputs
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Side-View Photo",
                            type="numpy",
                            sources=["upload", "clipboard"],
                        )

                        gr.Markdown("### Workstation Settings")
                        hours_slider = gr.Slider(
                            minimum=0.5, maximum=12, value=8, step=0.5,
                            label="Hours per day at workstation",
                        )

                        with gr.Row():
                            armrests_cb = gr.Checkbox(value=True, label="Armrests present")
                            lumbar_cb = gr.Checkbox(value=True, label="Lumbar support")

                        with gr.Accordion("Equipment Adjustability", open=False):
                            non_adj_chair = gr.Checkbox(label="Non-adjustable seat height")
                            non_adj_armrest = gr.Checkbox(label="Non-adjustable armrests")
                            non_adj_back = gr.Checkbox(label="Non-adjustable backrest")
                            non_adj_keyboard = gr.Checkbox(label="Non-adjustable keyboard tray")

                        with gr.Accordion("Additional Factors", open=False):
                            glare_cb = gr.Checkbox(label="Screen glare present")
                            no_doc_cb = gr.Checkbox(label="No document holder")
                            phone_dd = gr.Dropdown(
                                choices=["None", "Hands-free", "In hand", "Neck/Shoulder hold"],
                                value="None",
                                label="Phone usage",
                            )

                        with gr.Accordion("Manual Score Overrides (Optional)", open=False):
                            gr.Markdown("*Set to 0 to use auto-detected score*")
                            override_chair = gr.Number(value=0, label="Override Chair Score (1-10)", precision=0)
                            override_mp = gr.Number(value=0, label="Override Monitor+Phone Score (1-10)", precision=0)
                            override_km = gr.Number(value=0, label="Override Keyboard+Mouse Score (1-10)", precision=0)
                            override_just = gr.Textbox(label="Override Justification", placeholder="Reason for manual override...")

                        analyze_btn = gr.Button("Analyze Posture", variant="primary", size="lg")

                    # Right column: Results
                    with gr.Column(scale=1):
                        score_display = gr.HTML(label="ROSA Score")
                        annotated_output = gr.Image(label="Annotated Analysis", type="numpy")

                        with gr.Row(visible=False) as results_row:
                            pdf_btn = gr.Button("Export PDF Report", variant="secondary")

                        breakdown_md = gr.Markdown(label="Score Breakdown")
                        recommendations_md = gr.Markdown(label="Recommendations")

                # Wire up single image analysis
                analyze_btn.click(
                    fn=analyze_single,
                    inputs=[
                        image_input, hours_slider, armrests_cb, lumbar_cb,
                        non_adj_chair, non_adj_armrest, non_adj_back, non_adj_keyboard,
                        glare_cb, no_doc_cb, phone_dd,
                        override_chair, override_mp, override_km, override_just,
                    ],
                    outputs=[annotated_output, score_display, breakdown_md, recommendations_md, results_row],
                )

                # PDF export
                pdf_output = gr.File(label="Download PDF Report", visible=False)
                pdf_btn.click(
                    fn=export_pdf,
                    inputs=[
                        image_input, hours_slider, armrests_cb, lumbar_cb,
                        non_adj_chair, non_adj_armrest, non_adj_back, non_adj_keyboard,
                        glare_cb, no_doc_cb, phone_dd,
                        override_chair, override_mp, override_km, override_just,
                    ],
                    outputs=[pdf_output],
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=[pdf_output],
                )

            # ── Tab 2: Batch Analysis ──
            with gr.TabItem("Batch Analysis", id="batch"):
                gr.Markdown("### Upload Multiple Images or a ZIP File")

                batch_files = gr.File(
                    label="Upload Images (JPG, PNG) or ZIP",
                    file_count="multiple",
                    type="filepath",
                )

                with gr.Row():
                    batch_hours = gr.Slider(minimum=0.5, maximum=12, value=8, step=0.5, label="Hours/day")
                    batch_armrests = gr.Checkbox(value=True, label="Armrests present")
                    batch_lumbar = gr.Checkbox(value=True, label="Lumbar support")

                batch_btn = gr.Button("Analyze All Images", variant="primary", size="lg")

                batch_summary = gr.Markdown(label="Batch Results")
                batch_csv = gr.File(label="Download CSV Export")
                batch_status = gr.Textbox(label="Status", interactive=False)

                batch_btn.click(
                    fn=analyze_batch,
                    inputs=[batch_files, batch_hours, batch_armrests, batch_lumbar],
                    outputs=[batch_summary, batch_csv, batch_status],
                )

            # ── Tab 3: Photo Guide ──
            with gr.TabItem("Photo Guide", id="guide"):
                gr.Markdown(PHOTO_GUIDE)

            # ── Tab 4: About ──
            with gr.TabItem("About", id="about"):
                gr.Markdown("""
                ## About LocalROSA

                **LocalROSA** is an AI-powered tool that automates the Rapid Office Strain Assessment (ROSA)
                methodology developed by Michael Sonne, Diane L. Villalta, and David M. Andrews (2012).

                ### How It Works
                1. **Upload** a side-view photo of a seated worker
                2. **MediaPipe Pose** detects 33 body landmarks
                3. **Joint angles** are calculated (knee, elbow, trunk, neck, wrist)
                4. **Official ROSA tables** are applied to compute section scores
                5. **Grand ROSA Score** (1-10) is computed with risk classification

                ### ROSA Score Interpretation
                | Score | Risk Level | Action |
                |-------|-----------|--------|
                | 1-3 | Low (Green) | Posture is acceptable |
                | 4-5 | Moderate (Yellow) | Further investigation needed, changes may be required |
                | 6-10 | High (Red) | Immediate investigation and corrective action required |

                ### Scoring Sections
                - **Section A (Chair):** Seat height, pan depth, armrests, back support
                - **Section B (Monitor & Phone):** Screen position, phone usage
                - **Section C (Keyboard & Mouse):** Wrist posture, mouse position

                ### Technology
                - **Computer Vision:** MediaPipe Pose (heavy model) for landmark detection
                - **Image Processing:** OpenCV for preprocessing and annotation
                - **UI:** Gradio web framework
                - **100% Offline:** All processing runs locally on your machine

                ### Reference
                Sonne, M., Villalta, D. L., & Andrews, D. M. (2012). Development and evaluation of an
                office ergonomic risk checklist: ROSA - Rapid Office Strain Assessment.
                Applied Ergonomics, 43(1), 98-108.

                ### License
                This tool is for educational and professional ergonomic assessment purposes.
                Always validate automated scores with professional judgment.
                """)

    return app


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_results_dir()
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
