# LocalROSA - AI-Powered Rapid Office Strain Assessment Engine

A **100% local, offline** web application that computes official **ROSA scores (1–10)** from side-view photos of computer workstations using computer vision and the official ROSA methodology.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Mac%20%7C%20Linux-lightgrey.svg)

## What is ROSA?

The **Rapid Office Strain Assessment (ROSA)** is a validated ergonomic tool developed by [Sonne, Villalta & Andrews (2012)](https://doi.org/10.1016/j.apergo.2011.03.008) for evaluating office workstation risk. It produces a score from 1 (low risk) to 10 (high risk) based on:

- **Chair** – seat height, pan depth, armrests, back support
- **Monitor & Telephone** – screen position, phone usage
- **Keyboard & Mouse** – wrist posture, mouse position

LocalROSA automates this assessment using **MediaPipe Pose** for body landmark detection and **OpenCV** for angle measurement.

## Features

- **Single Image Analysis** – Upload one photo, get full ROSA breakdown
- **Batch Processing** – Analyze multiple images or ZIP files at once
- **Annotated Output** – Skeleton overlay with color-coded angle labels
- **PDF Reports** – Professional reports with images, scores, and recommendations
- **CSV Export** – Batch results in spreadsheet-ready format
- **Manual Overrides** – Adjust auto-detected scores with justification
- **Photo Guide** – Built-in tips for taking optimal assessment photos
- **Dark/Light Theme** – Professional Gradio interface with theme support
- **100% Offline** – No data ever leaves your machine

## Quick Start

### Prerequisites

- **Python 3.11+** (tested on 3.11, 3.12, 3.13)
- **pip** (Python package manager)
- ~500 MB disk space (including MediaPipe models)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mdyasirh/my-app.git
cd my-app

# 2. (Recommended) Create a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser to **http://127.0.0.1:7860**

That's it! The application runs entirely on your local machine.

## Usage Guide

### Single Image Analysis

1. Go to the **"Single Image Analysis"** tab
2. Upload a **side-view photo** of a seated worker
3. Adjust workstation settings (hours/day, armrests, lumbar support, etc.)
4. Click **"Analyze Posture"**
5. View the annotated image, ROSA score breakdown, and recommendations
6. Click **"Export PDF Report"** for a downloadable report

### Batch Analysis

1. Go to the **"Batch Analysis"** tab
2. Upload multiple images or a ZIP file
3. Set default workstation parameters
4. Click **"Analyze All Images"**
5. View summary statistics and download the CSV export

### Manual Overrides

For edge cases where auto-detection may be inaccurate:
1. Expand **"Manual Score Overrides"** in the sidebar
2. Enter corrected scores (1-10) for Chair, Monitor+Phone, or Keyboard+Mouse
3. Add a justification note
4. Re-analyze – overridden scores will be marked in the report

## How to Take Perfect Photos

For best accuracy:

| Requirement | Details |
|------------|---------|
| **Angle** | Directly from the side (90° to the front of the person) |
| **Distance** | 6-10 feet (2-3 meters) from subject |
| **Height** | Camera at waist/seat level |
| **Visibility** | Full body: head to feet, including chair and desk |
| **Lighting** | Even, bright lighting; avoid backlighting |
| **Posture** | Subject in natural working position |
| **Format** | JPG, PNG, BMP, TIFF, or WebP |
| **Resolution** | Minimum 640x480 (higher is better) |

## ROSA Score Interpretation

| Score | Risk Level | Color | Action Required |
|-------|-----------|-------|----------------|
| 1–3 | **Low** | Green | Posture is acceptable |
| 4–5 | **Moderate** | Yellow | Further investigation needed; changes may be beneficial |
| 6–10 | **High** | Red | Investigation and corrective action required immediately |

## Project Structure

```
my-app/
├── app.py                  # Main Gradio web application
├── pose_detector.py        # MediaPipe pose detection engine
├── rosa_calculator.py      # Complete ROSA scoring logic (all tables)
├── report_generator.py     # PDF and CSV report generation
├── utils.py                # Image processing and utility functions
├── build_exe.py            # PyInstaller standalone build script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── tests/
│   ├── __init__.py
│   └── test_rosa_calculator.py  # Unit tests (10 test cases)
├── results/                # Auto-created output directory
├── sample_images/          # Test image descriptions
│   └── README_SAMPLES.md
└── static/                 # Static assets
```

## ROSA Scoring Implementation

The scoring engine implements **100% of the official ROSA tables**:

### Section A: Chair
- **Seat Pan Height** → based on knee angle (90° ideal)
- **Seat Pan Depth** → 3" space behind knee
- **Armrests** → elbow at 90°, shoulder position
- **Back Support** → 95°–110° recline with lumbar
- Combined via **Chair Score Table** (8×8 matrix)
- Duration factor applied (-1/0/+1)

### Section B: Monitor & Telephone
- **Monitor** → eye level, arm's length, no twist/glare
- **Telephone** → neutral neck, hands-free
- Combined via **Monitor-Phone Table** (9×9 matrix)
- Duration factor applied

### Section C: Keyboard & Mouse
- **Keyboard** → wrist straight, shoulders relaxed
- **Mouse** → in-line with shoulder, no reaching
- Combined via **Keyboard-Mouse Table** (9×9 matrix)
- Duration factor applied

### Final Score
- **Peripherals/Monitor Score** = Sections B + C combined (9×9 table)
- **Grand ROSA Score** = Chair Score vs Peripherals Score (10×10 table)

All additive rules (+1 for non-adjustable equipment, glare, etc.) are implemented.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ -v --cov=. --cov-report=term-missing
```

## Building Standalone Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Run the build script
python build_exe.py

# The executable will be in dist/LocalROSA/
```

## Technical Details

- **Pose Detection**: MediaPipe Pose (model_complexity=2, heavy model)
- **Side-View Validation**: Compares shoulder/hip spread ratio to body height
- **Angle Calculation**: Vector math on 2D projected landmark coordinates
- **Image Preprocessing**: CLAHE contrast enhancement, auto-rotation from EXIF
- **Confidence Threshold**: Warns below 50%, full accuracy above 85%
- **PDF Generation**: fpdf2 (pure Python, no external dependencies)

## Limitations

- Requires **side-view photos** (front/back views will be flagged)
- Wrist deviation is estimated (no detailed hand landmarks in Pose model)
- Chair edge and desk detection are approximate
- Monitor distance is user-reported (cannot be measured from a single photo)
- Phone usage must be indicated by the user
- Best results with clear, well-lit photos showing full body

## Tips for Achieving Maximum Accuracy

1. **Always use side-view photos** taken from directly to the left or right
2. **Ensure full body visibility** from head to feet (including the chair)
3. **Good lighting is essential** – avoid dark rooms or harsh backlighting
4. **Subject should be in natural position** – not posed or adjusted
5. **Use the heavy model** (default) for best landmark detection
6. **Verify with manual override** for critical assessments
7. **Cross-reference** automated scores with visual inspection
8. **Take multiple photos** and compare results for consistency

## Reference

Sonne, M., Villalta, D. L., & Andrews, D. M. (2012). Development and evaluation of an office ergonomic risk checklist: ROSA – Rapid Office Strain Assessment. *Applied Ergonomics*, 43(1), 98–108. https://doi.org/10.1016/j.apergo.2011.03.008

## License

MIT License. This tool is for educational and professional ergonomic assessment purposes. Always validate automated scores with professional ergonomic judgment.
