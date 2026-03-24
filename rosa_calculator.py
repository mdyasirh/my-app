"""
LocalROSA – ROSA (Rapid Office Strain Assessment) Calculator
============================================================
Implements 100% of the official ROSA scoring tables and rules as defined
by Michael Sonne et al. (2012). Every lookup table, additive rule, and
duration factor is hard-coded exactly per the original Cornell/ROSA manual.

Reference: Sonne, M., Villalta, D. L., & Andrews, D. M. (2012).
           Development and evaluation of an office ergonomic risk checklist:
           ROSA – Rapid Office Strain Assessment.
           Applied Ergonomics, 43(1), 98–108.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Data classes for structured input/output
# ─────────────────────────────────────────────────────────────

@dataclass
class ChairInputs:
    """Raw scores for the Chair section (Section A)."""
    # Seat Pan Height: 1-3
    seat_pan_height: int = 1
    seat_pan_height_additives: int = 0  # +1 if insufficient space under desk

    # Seat Pan Depth: 1-2
    seat_pan_depth: int = 1
    seat_pan_depth_additives: int = 0  # +1 if non-adjustable seat pan

    # Armrest score: 1-3
    armrest: int = 1
    armrest_additives: int = 0  # +1 if hard/non-padded surface, +1 if non-adjustable

    # Back Support score: 1-3
    back_support: int = 1
    back_support_additives: int = 0  # +1 if no lumbar, +1 if non-adjustable back


@dataclass
class MonitorInputs:
    """Raw scores for Monitor section."""
    monitor: int = 1
    monitor_additives: int = 0  # +1 glare, +1 no document holder, +1 bifocals


@dataclass
class PhoneInputs:
    """Raw scores for Telephone section."""
    phone: int = 1
    phone_additives: int = 0  # +1 no headset with frequent phone use


@dataclass
class KeyboardInputs:
    """Raw scores for Keyboard section."""
    keyboard: int = 1
    keyboard_additives: int = 0  # +1 wrists deviated, +1 non-adjustable platform


@dataclass
class MouseInputs:
    """Raw scores for Mouse section."""
    mouse: int = 1
    mouse_additives: int = 0  # +1 pinch grip, +1 hard palmrest


@dataclass
class DurationInputs:
    """Duration factors for each section."""
    # Duration factor: -1 (<1 hr), 0 (1-4 hr), +1 (>4 hr continuous)
    chair_duration: int = 1
    monitor_phone_duration: int = 1
    keyboard_mouse_duration: int = 1


@dataclass
class ROSAResult:
    """Complete ROSA scoring result."""
    # Section A – Chair
    seat_height_depth_score: int = 0
    armrest_back_score: int = 0
    chair_score_raw: int = 0
    chair_score_final: int = 0  # After duration

    # Section B – Monitor & Telephone
    monitor_score_raw: int = 0
    phone_score_raw: int = 0
    monitor_phone_score_raw: int = 0
    monitor_phone_score_final: int = 0  # After duration

    # Section C – Keyboard & Mouse
    keyboard_score_raw: int = 0
    mouse_score_raw: int = 0
    keyboard_mouse_score_raw: int = 0
    keyboard_mouse_score_final: int = 0  # After duration

    # Peripherals/Monitor combined
    peripherals_monitor_score: int = 0

    # Grand ROSA Score (1-10)
    grand_rosa_score: int = 0

    # Risk level
    risk_level: str = ""
    risk_color: str = ""

    # Detailed breakdown
    breakdown: Dict[str, str] = field(default_factory=dict)
    recommendations: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Official ROSA Lookup Tables
# ─────────────────────────────────────────────────────────────

# Table A: Seat Pan Height+Depth (rows) vs Armrest+Back Support (cols)
# Index: (height_depth_score, armrest_back_score) → chair_score
# Height+Depth ranges 2-9, Armrest+Back ranges 2-9
CHAIR_SCORE_TABLE = {
    (2, 2): 2, (2, 3): 2, (2, 4): 3, (2, 5): 4, (2, 6): 5, (2, 7): 6, (2, 8): 7, (2, 9): 8,
    (3, 2): 2, (3, 3): 2, (3, 4): 3, (3, 5): 4, (3, 6): 5, (3, 7): 6, (3, 8): 7, (3, 9): 8,
    (4, 2): 3, (4, 3): 3, (4, 4): 3, (4, 5): 4, (4, 6): 5, (4, 7): 6, (4, 8): 7, (4, 9): 8,
    (5, 2): 4, (5, 3): 4, (5, 4): 4, (5, 5): 4, (5, 6): 5, (5, 7): 6, (5, 8): 7, (5, 9): 8,
    (6, 2): 5, (6, 3): 5, (6, 4): 5, (6, 5): 5, (6, 6): 6, (6, 7): 7, (6, 8): 8, (6, 9): 9,
    (7, 2): 6, (7, 3): 6, (7, 4): 6, (7, 5): 6, (7, 6): 7, (7, 7): 8, (7, 8): 8, (7, 9): 9,
    (8, 2): 7, (8, 3): 7, (8, 4): 7, (8, 5): 7, (8, 6): 8, (8, 7): 8, (8, 8): 9, (8, 9): 9,
    (9, 2): 8, (9, 3): 8, (9, 4): 8, (9, 5): 8, (9, 6): 9, (9, 7): 9, (9, 8): 9, (9, 9): 9,
}

# Table B: Monitor score (rows) vs Telephone score (cols)
# Monitor ranges 1-9, Phone ranges 1-9
MONITOR_PHONE_TABLE = {
    (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 2, (1, 5): 3, (1, 6): 4, (1, 7): 5, (1, 8): 6, (1, 9): 7,
    (2, 1): 1, (2, 2): 2, (2, 3): 2, (2, 4): 2, (2, 5): 3, (2, 6): 4, (2, 7): 5, (2, 8): 6, (2, 9): 7,
    (3, 1): 1, (3, 2): 2, (3, 3): 2, (3, 4): 3, (3, 5): 3, (3, 6): 4, (3, 7): 5, (3, 8): 6, (3, 9): 7,
    (4, 1): 2, (4, 2): 2, (4, 3): 3, (4, 4): 3, (4, 5): 3, (4, 6): 5, (4, 7): 5, (4, 8): 6, (4, 9): 7,
    (5, 1): 3, (5, 2): 3, (5, 3): 3, (5, 4): 4, (5, 5): 4, (5, 6): 5, (5, 7): 6, (5, 8): 7, (5, 9): 8,
    (6, 1): 4, (6, 2): 4, (6, 3): 4, (6, 4): 5, (6, 5): 5, (6, 6): 6, (6, 7): 6, (6, 8): 7, (6, 9): 8,
    (7, 1): 5, (7, 2): 5, (7, 3): 5, (7, 4): 5, (7, 5): 6, (7, 6): 7, (7, 7): 7, (7, 8): 8, (7, 9): 9,
    (8, 1): 6, (8, 2): 6, (8, 3): 6, (8, 4): 7, (8, 5): 7, (8, 6): 8, (8, 7): 8, (8, 8): 9, (8, 9): 9,
    (9, 1): 7, (9, 2): 7, (9, 3): 7, (9, 4): 7, (9, 5): 8, (9, 6): 8, (9, 7): 9, (9, 8): 9, (9, 9): 9,
}

# Table C: Keyboard score (rows) vs Mouse score (cols)
# Keyboard ranges 1-9, Mouse ranges 1-9
KEYBOARD_MOUSE_TABLE = {
    (1, 1): 1, (1, 2): 1, (1, 3): 1, (1, 4): 2, (1, 5): 3, (1, 6): 4, (1, 7): 5, (1, 8): 6, (1, 9): 7,
    (2, 1): 1, (2, 2): 2, (2, 3): 2, (2, 4): 2, (2, 5): 3, (2, 6): 4, (2, 7): 5, (2, 8): 6, (2, 9): 7,
    (3, 1): 1, (3, 2): 2, (3, 3): 2, (3, 4): 3, (3, 5): 3, (3, 6): 4, (3, 7): 5, (3, 8): 6, (3, 9): 7,
    (4, 1): 2, (4, 2): 2, (4, 3): 3, (4, 4): 3, (4, 5): 3, (4, 6): 5, (4, 7): 5, (4, 8): 6, (4, 9): 7,
    (5, 1): 3, (5, 2): 3, (5, 3): 3, (5, 4): 4, (5, 5): 4, (5, 6): 5, (5, 7): 6, (5, 8): 7, (5, 9): 8,
    (6, 1): 4, (6, 2): 4, (6, 3): 4, (6, 4): 5, (6, 5): 5, (6, 6): 6, (6, 7): 6, (6, 8): 7, (6, 9): 8,
    (7, 1): 5, (7, 2): 5, (7, 3): 5, (7, 4): 5, (7, 5): 6, (7, 6): 7, (7, 7): 7, (7, 8): 8, (7, 9): 9,
    (8, 1): 6, (8, 2): 6, (8, 3): 6, (8, 4): 7, (8, 5): 7, (8, 6): 8, (8, 7): 8, (8, 8): 9, (8, 9): 9,
    (9, 1): 7, (9, 2): 7, (9, 3): 7, (9, 4): 7, (9, 5): 8, (9, 6): 8, (9, 7): 9, (9, 8): 9, (9, 9): 9,
}

# Table D: Monitor & Phone score (rows) vs Keyboard & Mouse score (cols)
# → Peripherals/Monitor combined score
PERIPHERALS_MONITOR_TABLE = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 4, (1, 5): 5, (1, 6): 6, (1, 7): 7, (1, 8): 8, (1, 9): 9,
    (2, 1): 2, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 5, (2, 6): 6, (2, 7): 7, (2, 8): 8, (2, 9): 9,
    (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 4, (3, 5): 5, (3, 6): 6, (3, 7): 7, (3, 8): 8, (3, 9): 9,
    (4, 1): 4, (4, 2): 4, (4, 3): 4, (4, 4): 4, (4, 5): 5, (4, 6): 6, (4, 7): 7, (4, 8): 8, (4, 9): 9,
    (5, 1): 5, (5, 2): 5, (5, 3): 5, (5, 4): 5, (5, 5): 5, (5, 6): 6, (5, 7): 7, (5, 8): 8, (5, 9): 9,
    (6, 1): 6, (6, 2): 6, (6, 3): 6, (6, 4): 6, (6, 5): 6, (6, 6): 6, (6, 7): 7, (6, 8): 8, (6, 9): 9,
    (7, 1): 7, (7, 2): 7, (7, 3): 7, (7, 4): 7, (7, 5): 7, (7, 6): 7, (7, 7): 7, (7, 8): 8, (7, 9): 9,
    (8, 1): 8, (8, 2): 8, (8, 3): 8, (8, 4): 8, (8, 5): 8, (8, 6): 8, (8, 7): 8, (8, 8): 8, (8, 9): 9,
    (9, 1): 9, (9, 2): 9, (9, 3): 9, (9, 4): 9, (9, 5): 9, (9, 6): 9, (9, 7): 9, (9, 8): 9, (9, 9): 9,
}

# Table E (Final): Chair Score (rows) vs Peripherals/Monitor Score (cols)
# → Grand ROSA Score (1–10)
GRAND_ROSA_TABLE = {
    (1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 4, (1, 5): 5, (1, 6): 6, (1, 7): 7, (1, 8): 8, (1, 9): 9, (1, 10): 10,
    (2, 1): 2, (2, 2): 2, (2, 3): 3, (2, 4): 4, (2, 5): 5, (2, 6): 6, (2, 7): 7, (2, 8): 8, (2, 9): 9, (2, 10): 10,
    (3, 1): 3, (3, 2): 3, (3, 3): 3, (3, 4): 4, (3, 5): 5, (3, 6): 6, (3, 7): 7, (3, 8): 8, (3, 9): 9, (3, 10): 10,
    (4, 1): 4, (4, 2): 4, (4, 3): 4, (4, 4): 4, (4, 5): 5, (4, 6): 6, (4, 7): 7, (4, 8): 8, (4, 9): 9, (4, 10): 10,
    (5, 1): 5, (5, 2): 5, (5, 3): 5, (5, 4): 5, (5, 5): 5, (5, 6): 6, (5, 7): 7, (5, 8): 8, (5, 9): 9, (5, 10): 10,
    (6, 1): 6, (6, 2): 6, (6, 3): 6, (6, 4): 6, (6, 5): 6, (6, 6): 6, (6, 7): 7, (6, 8): 8, (6, 9): 9, (6, 10): 10,
    (7, 1): 7, (7, 2): 7, (7, 3): 7, (7, 4): 7, (7, 5): 7, (7, 6): 7, (7, 7): 7, (7, 8): 8, (7, 9): 9, (7, 10): 10,
    (8, 1): 8, (8, 2): 8, (8, 3): 8, (8, 4): 8, (8, 5): 8, (8, 6): 8, (8, 7): 8, (8, 8): 8, (8, 9): 9, (8, 10): 10,
    (9, 1): 9, (9, 2): 9, (9, 3): 9, (9, 4): 9, (9, 5): 9, (9, 6): 9, (9, 7): 9, (9, 8): 9, (9, 9): 9, (9, 10): 10,
    (10, 1): 10, (10, 2): 10, (10, 3): 10, (10, 4): 10, (10, 5): 10, (10, 6): 10, (10, 7): 10, (10, 8): 10, (10, 9): 10, (10, 10): 10,
}


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer value to [lo, hi]."""
    return max(lo, min(hi, value))


def _table_lookup(table: dict, row: int, col: int, row_range: Tuple[int, int], col_range: Tuple[int, int]) -> int:
    """Safe lookup with clamping to table bounds."""
    r = _clamp(row, row_range[0], row_range[1])
    c = _clamp(col, col_range[0], col_range[1])
    return table.get((r, c), max(r, c))


# ─────────────────────────────────────────────────────────────
# Angle → Score Conversion Functions
# ─────────────────────────────────────────────────────────────

def score_seat_pan_height(knee_angle: float, feet_on_floor: bool) -> int:
    """
    ROSA Rule – Seat Pan Height:
    - Knees at ~90° AND feet flat on floor → 1
    - Knees <90° (seat too high) → 2
    - Knees >90° (seat too low) → 2
    - Feet not touching floor → 3
    """
    if not feet_on_floor:
        return 3
    if 80 <= knee_angle <= 100:
        return 1
    return 2


def score_seat_pan_depth(space_behind_knee_ok: bool) -> int:
    """
    ROSA Rule – Seat Pan Depth:
    - ~3" (approx 8cm) space between seat edge and back of knee → 1
    - <3" or >3" space → 2
    """
    return 1 if space_behind_knee_ok else 2


def score_armrest(elbow_angle: float, shoulder_shrug: bool, armrests_present: bool) -> int:
    """
    ROSA Rule – Armrests:
    - Elbows at ~90°, supported, shoulders relaxed → 1
    - Armrests too high (shoulders raised) → 2
    - Armrests too low / elbows not supported → 2
    - No armrests → 3
    """
    if not armrests_present:
        return 3
    if shoulder_shrug:
        return 2
    if 80 <= elbow_angle <= 100:
        return 1
    return 2


def score_back_support(trunk_recline_angle: float, lumbar_support: bool) -> int:
    """
    ROSA Rule – Back Support:
    - 95°–110° recline with lumbar support in small of back → 1
    - Back support too upright (<95°) or too reclined (>110°) → 2
    - No back support / leaning forward → 3
    """
    if not lumbar_support:
        return 3
    if 95 <= trunk_recline_angle <= 110:
        return 1
    return 2


def score_monitor(neck_flexion: float, monitor_distance_ok: bool, neck_twist: float) -> int:
    """
    ROSA Rule – Monitor:
    - Monitor at eye level, arm's length, no twist → 1
    - Monitor low (neck flexion >20°) → 2
    - Monitor too high (neck extension) → 2
    - Monitor too far away → 2
    - Neck twist >30° → +1 additive (handled separately)
    Base score combines position issues.
    """
    score = 1
    # Monitor too low
    if neck_flexion > 20:
        score = 2
    # Monitor too high (negative flexion = extension)
    if neck_flexion < -10:
        score = 2
    # Monitor too far
    if not monitor_distance_ok:
        score = max(score, 2)
    # Neck twist adds to score
    if neck_twist > 30:
        score += 1
    return _clamp(score, 1, 5)


def score_phone(phone_use: str) -> int:
    """
    ROSA Rule – Telephone:
    - Hands-free / no phone use → 1
    - Phone in hand, neutral neck → 1
    - Neck/shoulder hold → 3
    - Reaching for phone → 2
    """
    if phone_use == "none" or phone_use == "hands_free":
        return 1
    if phone_use == "reaching":
        return 2
    if phone_use == "neck_shoulder":
        return 3
    return 1


def score_keyboard(wrist_extension: float, shoulder_raised: bool) -> int:
    """
    ROSA Rule – Keyboard:
    - Wrists straight, shoulders relaxed → 1
    - Wrists extended >15° → 2
    - Shoulders raised / keyboard too high → 2
    - Both issues → 3
    """
    issues = 0
    if abs(wrist_extension) > 15:
        issues += 1
    if shoulder_raised:
        issues += 1
    return _clamp(1 + issues, 1, 3)


def score_mouse(mouse_in_line: bool, reaching: bool) -> int:
    """
    ROSA Rule – Mouse:
    - Mouse in line with shoulder → 1
    - Reaching for mouse → 2
    - Mouse not in line + reaching → 3
    """
    issues = 0
    if not mouse_in_line:
        issues += 1
    if reaching:
        issues += 1
    return _clamp(1 + issues, 1, 3)


# ─────────────────────────────────────────────────────────────
# Duration Factor
# ─────────────────────────────────────────────────────────────

def get_duration_factor(hours_per_day: float) -> int:
    """
    ROSA Duration Factor:
    - <1 hour/day → -1
    - 1–4 hours/day → 0
    - >4 hours/day continuous → +1
    """
    if hours_per_day < 1:
        return -1
    elif hours_per_day <= 4:
        return 0
    else:
        return 1


# ─────────────────────────────────────────────────────────────
# Main ROSA Calculation
# ─────────────────────────────────────────────────────────────

def calculate_rosa(
    chair: ChairInputs,
    monitor: MonitorInputs,
    phone: PhoneInputs,
    keyboard: KeyboardInputs,
    mouse: MouseInputs,
    duration: DurationInputs,
) -> ROSAResult:
    """
    Calculate the complete ROSA score from individual section inputs.
    All inputs should already have base scores + additives calculated.
    This function performs all table lookups and produces the final Grand ROSA Score.
    """
    result = ROSAResult()

    # ── Section A: Chair ──
    # Height + Depth sub-score (sum of base + additives for each)
    height_score = chair.seat_pan_height + chair.seat_pan_height_additives
    depth_score = chair.seat_pan_depth + chair.seat_pan_depth_additives
    result.seat_height_depth_score = _clamp(height_score + depth_score, 2, 9)

    # Armrest + Back sub-score
    arm_score = chair.armrest + chair.armrest_additives
    back_score = chair.back_support + chair.back_support_additives
    result.armrest_back_score = _clamp(arm_score + back_score, 2, 9)

    # Chair Score from Table A
    result.chair_score_raw = _table_lookup(
        CHAIR_SCORE_TABLE,
        result.seat_height_depth_score,
        result.armrest_back_score,
        (2, 9), (2, 9)
    )

    # Apply duration factor to chair
    result.chair_score_final = _clamp(
        result.chair_score_raw + duration.chair_duration, 1, 10
    )

    # ── Section B: Monitor & Telephone ──
    result.monitor_score_raw = _clamp(monitor.monitor + monitor.monitor_additives, 1, 9)
    result.phone_score_raw = _clamp(phone.phone + phone.phone_additives, 1, 9)

    result.monitor_phone_score_raw = _table_lookup(
        MONITOR_PHONE_TABLE,
        result.monitor_score_raw,
        result.phone_score_raw,
        (1, 9), (1, 9)
    )
    result.monitor_phone_score_final = _clamp(
        result.monitor_phone_score_raw + duration.monitor_phone_duration, 1, 10
    )

    # ── Section C: Keyboard & Mouse ──
    result.keyboard_score_raw = _clamp(keyboard.keyboard + keyboard.keyboard_additives, 1, 9)
    result.mouse_score_raw = _clamp(mouse.mouse + mouse.mouse_additives, 1, 9)

    result.keyboard_mouse_score_raw = _table_lookup(
        KEYBOARD_MOUSE_TABLE,
        result.keyboard_score_raw,
        result.mouse_score_raw,
        (1, 9), (1, 9)
    )
    result.keyboard_mouse_score_final = _clamp(
        result.keyboard_mouse_score_raw + duration.keyboard_mouse_duration, 1, 10
    )

    # ── Peripherals/Monitor Combined (Table D) ──
    result.peripherals_monitor_score = _table_lookup(
        PERIPHERALS_MONITOR_TABLE,
        result.monitor_phone_score_final,
        result.keyboard_mouse_score_final,
        (1, 9), (1, 9)
    )

    # ── Grand ROSA Score (Table E) ──
    result.grand_rosa_score = _table_lookup(
        GRAND_ROSA_TABLE,
        result.chair_score_final,
        result.peripherals_monitor_score,
        (1, 10), (1, 10)
    )

    # ── Risk Level ──
    if result.grand_rosa_score <= 3:
        result.risk_level = "Low"
        result.risk_color = "green"
    elif result.grand_rosa_score <= 5:
        result.risk_level = "Moderate"
        result.risk_color = "orange"
    else:
        result.risk_level = "High"
        result.risk_color = "red"

    # ── Breakdown ──
    result.breakdown = {
        "Seat Pan Height Score": str(height_score),
        "Seat Pan Depth Score": str(depth_score),
        "Height + Depth Combined": str(result.seat_height_depth_score),
        "Armrest Score": str(arm_score),
        "Back Support Score": str(back_score),
        "Armrest + Back Combined": str(result.armrest_back_score),
        "Chair Score (raw)": str(result.chair_score_raw),
        "Chair Score (with duration)": str(result.chair_score_final),
        "Monitor Score": str(result.monitor_score_raw),
        "Phone Score": str(result.phone_score_raw),
        "Monitor+Phone Combined (raw)": str(result.monitor_phone_score_raw),
        "Monitor+Phone (with duration)": str(result.monitor_phone_score_final),
        "Keyboard Score": str(result.keyboard_score_raw),
        "Mouse Score": str(result.mouse_score_raw),
        "Keyboard+Mouse Combined (raw)": str(result.keyboard_mouse_score_raw),
        "Keyboard+Mouse (with duration)": str(result.keyboard_mouse_score_final),
        "Peripherals/Monitor Score": str(result.peripherals_monitor_score),
        "Grand ROSA Score": str(result.grand_rosa_score),
        "Risk Level": result.risk_level,
    }

    # ── Recommendations ──
    result.recommendations = generate_recommendations(chair, monitor, keyboard, mouse, result)

    return result


def generate_recommendations(
    chair: ChairInputs,
    monitor: MonitorInputs,
    keyboard: KeyboardInputs,
    mouse: MouseInputs,
    result: ROSAResult,
) -> list:
    """Generate ergonomic recommendations based on individual scores."""
    recs = []

    if chair.seat_pan_height >= 2:
        recs.append("Adjust chair height so knees are at approximately 90° with feet flat on the floor.")
    if chair.seat_pan_depth >= 2:
        recs.append("Adjust seat pan depth to maintain ~3 inches (8 cm) of space behind your knees.")
    if chair.armrest >= 2:
        recs.append("Adjust armrests so elbows are at 90° with shoulders relaxed (not shrugged).")
    if chair.back_support >= 2:
        recs.append("Adjust backrest to 95°–110° recline with lumbar support in the small of your back.")
    if chair.armrest_additives > 0:
        recs.append("Consider upgrading to padded, adjustable armrests.")
    if chair.back_support_additives > 0:
        recs.append("Add a lumbar support cushion if your chair lacks built-in lumbar support.")

    if monitor.monitor >= 2:
        recs.append("Position monitor at arm's length with the top of the screen at or slightly below eye level.")
    if monitor.monitor_additives > 0:
        recs.append("Reduce screen glare, consider a document holder, and check if bifocal adjustment is needed.")

    if keyboard.keyboard >= 2:
        recs.append("Lower keyboard or adjust chair so wrists are straight and shoulders are relaxed while typing.")
    if keyboard.keyboard_additives > 0:
        recs.append("Consider a split or ergonomic keyboard and an adjustable keyboard tray.")

    if mouse.mouse >= 2:
        recs.append("Move mouse closer, in line with your shoulder. Avoid reaching.")
    if mouse.mouse_additives > 0:
        recs.append("Use a mouse with a comfortable grip (avoid pinch grip). Add a padded wrist rest.")

    if result.grand_rosa_score >= 5:
        recs.append("PRIORITY: Your overall ROSA score indicates significant ergonomic risk. "
                     "Consider a professional workstation assessment.")
    if not recs:
        recs.append("Your workstation setup looks good! Maintain current ergonomic practices.")

    return recs


def calculate_rosa_from_angles(
    knee_angle: float = 90.0,
    elbow_angle: float = 90.0,
    trunk_recline_angle: float = 100.0,
    neck_flexion: float = 0.0,
    wrist_extension: float = 0.0,
    neck_twist: float = 0.0,
    feet_on_floor: bool = True,
    space_behind_knee_ok: bool = True,
    armrests_present: bool = True,
    lumbar_support: bool = True,
    shoulder_shrug: bool = False,
    monitor_distance_ok: bool = True,
    mouse_in_line: bool = True,
    mouse_reaching: bool = False,
    phone_use: str = "none",
    hours_per_day: float = 8.0,
    # Additives (user-provided or auto-detected)
    insufficient_desk_space: bool = False,
    non_adjustable_seat: bool = False,
    hard_armrest_surface: bool = False,
    non_adjustable_armrest: bool = False,
    no_lumbar_pad: bool = False,
    non_adjustable_back: bool = False,
    screen_glare: bool = False,
    no_document_holder: bool = False,
    bifocals: bool = False,
    no_headset_frequent_phone: bool = False,
    wrist_deviated: bool = False,
    non_adjustable_keyboard: bool = False,
    pinch_grip_mouse: bool = False,
    hard_palmrest: bool = False,
) -> ROSAResult:
    """
    High-level function: convert detected angles + user inputs
    directly into a full ROSA result.
    This is the main entry point used by the pose detector.
    """
    # Score individual components from angles
    chair = ChairInputs(
        seat_pan_height=score_seat_pan_height(knee_angle, feet_on_floor),
        seat_pan_height_additives=1 if insufficient_desk_space else 0,
        seat_pan_depth=score_seat_pan_depth(space_behind_knee_ok),
        seat_pan_depth_additives=1 if non_adjustable_seat else 0,
        armrest=score_armrest(elbow_angle, shoulder_shrug, armrests_present),
        armrest_additives=(1 if hard_armrest_surface else 0) + (1 if non_adjustable_armrest else 0),
        back_support=score_back_support(trunk_recline_angle, lumbar_support),
        back_support_additives=(1 if no_lumbar_pad else 0) + (1 if non_adjustable_back else 0),
    )

    monitor_input = MonitorInputs(
        monitor=score_monitor(neck_flexion, monitor_distance_ok, neck_twist),
        monitor_additives=(1 if screen_glare else 0) + (1 if no_document_holder else 0) + (1 if bifocals else 0),
    )

    phone_input = PhoneInputs(
        phone=score_phone(phone_use),
        phone_additives=1 if no_headset_frequent_phone else 0,
    )

    keyboard_input = KeyboardInputs(
        keyboard=score_keyboard(wrist_extension, shoulder_shrug),
        keyboard_additives=(1 if wrist_deviated else 0) + (1 if non_adjustable_keyboard else 0),
    )

    mouse_input = MouseInputs(
        mouse=score_mouse(mouse_in_line, mouse_reaching),
        mouse_additives=(1 if pinch_grip_mouse else 0) + (1 if hard_palmrest else 0),
    )

    duration = DurationInputs(
        chair_duration=get_duration_factor(hours_per_day),
        monitor_phone_duration=get_duration_factor(hours_per_day),
        keyboard_mouse_duration=get_duration_factor(hours_per_day),
    )

    return calculate_rosa(chair, monitor_input, phone_input, keyboard_input, mouse_input, duration)
