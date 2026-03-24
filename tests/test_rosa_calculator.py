"""
Unit tests for ROSA scoring calculator.
Tests every scoring table with known angle → expected score mappings.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosa_calculator import (
    calculate_rosa,
    calculate_rosa_from_angles,
    score_seat_pan_height,
    score_seat_pan_depth,
    score_armrest,
    score_back_support,
    score_monitor,
    score_phone,
    score_keyboard,
    score_mouse,
    get_duration_factor,
    ChairInputs,
    MonitorInputs,
    PhoneInputs,
    KeyboardInputs,
    MouseInputs,
    DurationInputs,
    CHAIR_SCORE_TABLE,
    MONITOR_PHONE_TABLE,
    KEYBOARD_MOUSE_TABLE,
    PERIPHERALS_MONITOR_TABLE,
    GRAND_ROSA_TABLE,
)


# ─────────────────────────────────────────────────────────────
# Test Individual Scoring Functions
# ─────────────────────────────────────────────────────────────

class TestSeatPanHeight:
    """Tests for ROSA seat pan height scoring."""

    def test_ideal_90_degrees(self):
        assert score_seat_pan_height(90.0, feet_on_floor=True) == 1

    def test_acceptable_range(self):
        assert score_seat_pan_height(85.0, feet_on_floor=True) == 1
        assert score_seat_pan_height(95.0, feet_on_floor=True) == 1

    def test_too_high_seat(self):
        """Knee angle <80° means seat is too high."""
        assert score_seat_pan_height(70.0, feet_on_floor=True) == 2

    def test_too_low_seat(self):
        """Knee angle >100° means seat is too low."""
        assert score_seat_pan_height(110.0, feet_on_floor=True) == 2

    def test_feet_not_on_floor(self):
        """Feet dangling → score 3 regardless of angle."""
        assert score_seat_pan_height(90.0, feet_on_floor=False) == 3
        assert score_seat_pan_height(70.0, feet_on_floor=False) == 3


class TestSeatPanDepth:
    def test_good_space(self):
        assert score_seat_pan_depth(space_behind_knee_ok=True) == 1

    def test_bad_space(self):
        assert score_seat_pan_depth(space_behind_knee_ok=False) == 2


class TestArmrest:
    def test_ideal(self):
        assert score_armrest(90.0, shoulder_shrug=False, armrests_present=True) == 1

    def test_no_armrests(self):
        assert score_armrest(90.0, shoulder_shrug=False, armrests_present=False) == 3

    def test_shoulder_shrug(self):
        assert score_armrest(90.0, shoulder_shrug=True, armrests_present=True) == 2

    def test_bad_angle(self):
        assert score_armrest(120.0, shoulder_shrug=False, armrests_present=True) == 2


class TestBackSupport:
    def test_ideal(self):
        assert score_back_support(100.0, lumbar_support=True) == 1

    def test_no_lumbar(self):
        assert score_back_support(100.0, lumbar_support=False) == 3

    def test_too_upright(self):
        assert score_back_support(85.0, lumbar_support=True) == 2

    def test_too_reclined(self):
        assert score_back_support(120.0, lumbar_support=True) == 2


class TestMonitor:
    def test_ideal(self):
        assert score_monitor(0.0, monitor_distance_ok=True, neck_twist=0.0) == 1

    def test_too_low(self):
        assert score_monitor(30.0, monitor_distance_ok=True, neck_twist=0.0) == 2

    def test_too_high(self):
        assert score_monitor(-15.0, monitor_distance_ok=True, neck_twist=0.0) == 2

    def test_with_twist(self):
        assert score_monitor(0.0, monitor_distance_ok=True, neck_twist=40.0) == 2

    def test_too_far(self):
        assert score_monitor(0.0, monitor_distance_ok=False, neck_twist=0.0) == 2


class TestPhone:
    def test_none(self):
        assert score_phone("none") == 1

    def test_hands_free(self):
        assert score_phone("hands_free") == 1

    def test_reaching(self):
        assert score_phone("reaching") == 2

    def test_neck_shoulder(self):
        assert score_phone("neck_shoulder") == 3


class TestKeyboard:
    def test_ideal(self):
        assert score_keyboard(0.0, shoulder_raised=False) == 1

    def test_wrist_extended(self):
        assert score_keyboard(20.0, shoulder_raised=False) == 2

    def test_shoulder_raised(self):
        assert score_keyboard(0.0, shoulder_raised=True) == 2

    def test_both_issues(self):
        assert score_keyboard(20.0, shoulder_raised=True) == 3


class TestMouse:
    def test_ideal(self):
        assert score_mouse(mouse_in_line=True, reaching=False) == 1

    def test_reaching(self):
        assert score_mouse(mouse_in_line=True, reaching=True) == 2

    def test_not_in_line(self):
        assert score_mouse(mouse_in_line=False, reaching=False) == 2

    def test_both_issues(self):
        assert score_mouse(mouse_in_line=False, reaching=True) == 3


class TestDuration:
    def test_short(self):
        assert get_duration_factor(0.5) == -1

    def test_medium(self):
        assert get_duration_factor(2.0) == 0

    def test_long(self):
        assert get_duration_factor(8.0) == 1

    def test_boundary_low(self):
        assert get_duration_factor(1.0) == 0

    def test_boundary_high(self):
        assert get_duration_factor(4.0) == 0


# ─────────────────────────────────────────────────────────────
# Test Lookup Tables
# ─────────────────────────────────────────────────────────────

class TestChairScoreTable:
    """Verify specific entries in the Chair Score lookup table."""

    def test_min(self):
        assert CHAIR_SCORE_TABLE[(2, 2)] == 2

    def test_max(self):
        assert CHAIR_SCORE_TABLE[(9, 9)] == 9

    def test_mid(self):
        assert CHAIR_SCORE_TABLE[(5, 5)] == 4

    def test_asymmetric(self):
        assert CHAIR_SCORE_TABLE[(3, 7)] == 6


class TestGrandRosaTable:
    def test_min(self):
        assert GRAND_ROSA_TABLE[(1, 1)] == 1

    def test_max(self):
        assert GRAND_ROSA_TABLE[(10, 10)] == 10

    def test_mid(self):
        assert GRAND_ROSA_TABLE[(5, 5)] == 5


# ─────────────────────────────────────────────────────────────
# Integration Tests: Full ROSA Calculation from Angles
# ─────────────────────────────────────────────────────────────

class TestFullROSACalculation:
    """
    End-to-end tests with known angles and expected ROSA scores.
    These represent standard test postures.
    """

    def test_perfect_posture(self):
        """
        Test Case 1: Ideal ergonomic setup.
        All angles optimal, full-day use.
        Expected: Low risk (score ~3-4 due to full-day duration).
        """
        result = calculate_rosa_from_angles(
            knee_angle=90, elbow_angle=90, trunk_recline_angle=100,
            neck_flexion=0, wrist_extension=0,
            feet_on_floor=True, space_behind_knee_ok=True,
            armrests_present=True, lumbar_support=True,
            shoulder_shrug=False, monitor_distance_ok=True,
            mouse_in_line=True, mouse_reaching=False,
            phone_use="none", hours_per_day=8.0,
        )
        assert result.grand_rosa_score <= 4
        assert result.risk_level in ("Low", "Moderate")

    def test_poor_chair_posture(self):
        """
        Test Case 2: Bad chair setup - knees too acute, no lumbar, no armrests.
        Expected: High risk.
        """
        result = calculate_rosa_from_angles(
            knee_angle=70, elbow_angle=120, trunk_recline_angle=80,
            feet_on_floor=True, space_behind_knee_ok=False,
            armrests_present=False, lumbar_support=False,
            shoulder_shrug=True, hours_per_day=8.0,
        )
        assert result.grand_rosa_score >= 5
        assert result.risk_level in ("Moderate", "High")

    def test_monitor_issues(self):
        """
        Test Case 3: Poor monitor position with neck flexion.
        """
        result = calculate_rosa_from_angles(
            knee_angle=90, elbow_angle=90, trunk_recline_angle=100,
            neck_flexion=35, wrist_extension=0,
            monitor_distance_ok=False,
            screen_glare=True, no_document_holder=True,
            hours_per_day=8.0,
        )
        assert result.monitor_score_raw >= 2

    def test_keyboard_mouse_issues(self):
        """
        Test Case 4: Poor keyboard/mouse setup.
        """
        result = calculate_rosa_from_angles(
            knee_angle=90, elbow_angle=90, trunk_recline_angle=100,
            wrist_extension=25, shoulder_shrug=True,
            mouse_in_line=False, mouse_reaching=True,
            pinch_grip_mouse=True, non_adjustable_keyboard=True,
            hours_per_day=8.0,
        )
        assert result.keyboard_score_raw >= 2
        assert result.mouse_score_raw >= 2

    def test_short_duration_reduces_score(self):
        """
        Test Case 5: Same posture but <1 hour/day should reduce scores.
        """
        result_long = calculate_rosa_from_angles(
            knee_angle=70, elbow_angle=120, trunk_recline_angle=80,
            hours_per_day=8.0,
        )
        result_short = calculate_rosa_from_angles(
            knee_angle=70, elbow_angle=120, trunk_recline_angle=80,
            hours_per_day=0.5,
        )
        assert result_short.grand_rosa_score <= result_long.grand_rosa_score

    def test_worst_case(self):
        """
        Test Case 6: Everything wrong - should produce high score.
        """
        result = calculate_rosa_from_angles(
            knee_angle=60, elbow_angle=140, trunk_recline_angle=75,
            neck_flexion=40, wrist_extension=25,
            feet_on_floor=False, space_behind_knee_ok=False,
            armrests_present=False, lumbar_support=False,
            shoulder_shrug=True, monitor_distance_ok=False,
            mouse_in_line=False, mouse_reaching=True,
            phone_use="neck_shoulder",
            hours_per_day=10.0,
            insufficient_desk_space=True, non_adjustable_seat=True,
            hard_armrest_surface=True, non_adjustable_armrest=True,
            no_lumbar_pad=True, non_adjustable_back=True,
            screen_glare=True, no_document_holder=True,
            no_headset_frequent_phone=True,
            wrist_deviated=True, non_adjustable_keyboard=True,
            pinch_grip_mouse=True, hard_palmrest=True,
        )
        assert result.grand_rosa_score >= 7
        assert result.risk_level == "High"

    def test_moderate_risk(self):
        """
        Test Case 7: Moderately poor posture.
        """
        result = calculate_rosa_from_angles(
            knee_angle=105, elbow_angle=110, trunk_recline_angle=115,
            neck_flexion=15, wrist_extension=10,
            armrests_present=True, lumbar_support=True,
            hours_per_day=6.0,
        )
        assert 3 <= result.grand_rosa_score <= 7

    def test_result_has_recommendations(self):
        """
        Test Case 8: Verify recommendations are generated.
        """
        result = calculate_rosa_from_angles(
            knee_angle=70, elbow_angle=120, trunk_recline_angle=80,
            lumbar_support=False, armrests_present=False,
            hours_per_day=8.0,
        )
        assert len(result.recommendations) > 0

    def test_result_has_breakdown(self):
        """Verify complete breakdown is present."""
        result = calculate_rosa_from_angles()
        assert "Grand ROSA Score" in result.breakdown
        assert "Risk Level" in result.breakdown

    def test_risk_colors(self):
        """Verify risk level classification."""
        low = calculate_rosa_from_angles(hours_per_day=0.5)
        assert low.risk_level == "Low"
        assert low.risk_color == "green"


# ─────────────────────────────────────────────────────────────
# Table Completeness Tests
# ─────────────────────────────────────────────────────────────

class TestTableCompleteness:
    """Ensure all lookup tables have complete entries."""

    def test_chair_table_complete(self):
        for r in range(2, 10):
            for c in range(2, 10):
                assert (r, c) in CHAIR_SCORE_TABLE

    def test_monitor_phone_table_complete(self):
        for r in range(1, 10):
            for c in range(1, 10):
                assert (r, c) in MONITOR_PHONE_TABLE

    def test_keyboard_mouse_table_complete(self):
        for r in range(1, 10):
            for c in range(1, 10):
                assert (r, c) in KEYBOARD_MOUSE_TABLE

    def test_peripherals_table_complete(self):
        for r in range(1, 10):
            for c in range(1, 10):
                assert (r, c) in PERIPHERALS_MONITOR_TABLE

    def test_grand_rosa_table_complete(self):
        for r in range(1, 11):
            for c in range(1, 11):
                assert (r, c) in GRAND_ROSA_TABLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
