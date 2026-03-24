"""
LocalROSA – Utility Functions
==============================
Image processing, file handling, and helper utilities.
"""

import io
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
RESULTS_DIR = Path("results")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_results_dir() -> Path:
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def load_image(file_path: str) -> Optional[np.ndarray]:
    """
    Load an image from file path with auto-rotation handling.

    Args:
        file_path: Path to the image file.

    Returns:
        BGR image as numpy array, or None if loading fails.
    """
    try:
        # Read with OpenCV
        image = cv2.imread(file_path)
        if image is None:
            # Try reading with numpy for edge cases
            with open(file_path, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if image is None:
            logger.error(f"Failed to load image: {file_path}")
            return None

        # Auto-rotate based on EXIF orientation
        image = auto_rotate_image(image, file_path)

        return image
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        return None


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Load image from bytes (for Gradio file upload)."""
    try:
        data = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Error loading image from bytes: {e}")
        return None


def auto_rotate_image(image: np.ndarray, file_path: str) -> np.ndarray:
    """
    Auto-rotate image based on EXIF orientation tag.
    Many phone cameras store images with rotation metadata.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import Tags as ExifTags

        pil_image = Image.open(file_path)
        exif = pil_image._getexif()

        if exif:
            # Find the orientation tag
            orientation_key = None
            for tag_id, tag_name in ExifTags.__dict__.items():
                if tag_name == "Orientation":
                    orientation_key = tag_id
                    break

            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif orientation == 6:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 8:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except ImportError:
        pass  # PIL not available, skip rotation
    except Exception:
        pass  # EXIF data not available or corrupt

    return image


def save_annotated_image(image: np.ndarray, original_name: str, suffix: str = "_annotated") -> str:
    """
    Save an annotated image to the results directory.

    Returns:
        Path to the saved image.
    """
    results_dir = ensure_results_dir()
    name = Path(original_name).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{name}{suffix}_{timestamp}.png"
    cv2.imwrite(str(output_path), image)
    logger.info(f"Saved annotated image: {output_path}")
    return str(output_path)


def extract_zip(zip_path: str, extract_to: Optional[str] = None) -> List[str]:
    """
    Extract images from a ZIP file.

    Returns:
        List of extracted image file paths.
    """
    if extract_to is None:
        extract_to = tempfile.mkdtemp(prefix="localrosa_")

    image_files = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                ext = Path(name).suffix.lower()
                if ext in SUPPORTED_FORMATS:
                    zf.extract(name, extract_to)
                    image_files.append(os.path.join(extract_to, name))
    except Exception as e:
        logger.error(f"Error extracting ZIP: {e}")

    return image_files


def get_risk_color_hex(risk_level: str) -> str:
    """Get hex color for risk level (for UI display)."""
    colors = {
        "Low": "#28a745",
        "Moderate": "#ffc107",
        "High": "#dc3545",
    }
    return colors.get(risk_level, "#6c757d")


def get_risk_emoji(risk_level: str) -> str:
    """Get indicator for risk level."""
    indicators = {
        "Low": "[LOW]",
        "Moderate": "[MODERATE]",
        "High": "[HIGH]",
    }
    return indicators.get(risk_level, "[?]")


def format_score_table(breakdown: dict) -> str:
    """Format ROSA breakdown as a readable text table."""
    lines = ["=" * 45]
    lines.append("       ROSA Score Breakdown")
    lines.append("=" * 45)

    for key, value in breakdown.items():
        lines.append(f"  {key:.<35} {value}")

    lines.append("=" * 45)
    return "\n".join(lines)


def validate_image_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a supported image format and is readable.

    Returns:
        (is_valid, message)
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {path.suffix}. Supported: {', '.join(SUPPORTED_FORMATS)}"

    # Try to read the image
    img = cv2.imread(str(path))
    if img is None:
        return False, f"Cannot read image file: {file_path}"

    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return False, f"Image too small ({w}x{h}). Minimum 100x100 pixels."

    return True, "OK"


def create_timestamp_folder() -> Path:
    """Create a timestamped folder for batch results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = RESULTS_DIR / f"batch_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
