"""
LocalROSA – Pose Detection Engine
==================================
Uses MediaPipe Pose to detect body landmarks from side-view photos,
extract joint angles, validate side-view orientation, and prepare
inputs for the ROSA scoring calculator.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe Pose landmark indices
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LM = mp.solutions.pose.PoseLandmark

# Key landmarks for ROSA assessment
LANDMARKS_OF_INTEREST = {
    "left_ear": LM.LEFT_EAR,
    "right_ear": LM.RIGHT_EAR,
    "left_eye": LM.LEFT_EYE,
    "right_eye": LM.RIGHT_EYE,
    "left_shoulder": LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_elbow": LM.LEFT_ELBOW,
    "right_elbow": LM.RIGHT_ELBOW,
    "left_wrist": LM.LEFT_WRIST,
    "right_wrist": LM.RIGHT_WRIST,
    "left_hip": LM.LEFT_HIP,
    "right_hip": LM.RIGHT_HIP,
    "left_knee": LM.LEFT_KNEE,
    "right_knee": LM.RIGHT_KNEE,
    "left_ankle": LM.LEFT_ANKLE,
    "right_ankle": LM.RIGHT_ANKLE,
    "nose": LM.NOSE,
}


@dataclass
class PoseAngles:
    """Extracted angles from pose detection."""
    knee_angle: float = 90.0
    elbow_angle: float = 90.0
    trunk_recline_angle: float = 100.0
    neck_flexion: float = 0.0
    wrist_extension: float = 0.0
    shoulder_shrug: bool = False
    side_view_confidence: float = 0.0
    is_valid_side_view: bool = True
    feet_on_floor: bool = True
    space_behind_knee_ok: bool = True
    detection_confidence: float = 0.0
    detected_side: str = "right"  # which side is facing camera
    landmarks: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class PoseDetector:
    """
    MediaPipe-based pose detector optimized for seated side-view analysis.
    Uses the heavy pose model for maximum accuracy.
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # 0=lite, 1=full, 2=heavy (most accurate)
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.min_confidence = 0.5  # Minimum per-landmark confidence

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pre-process image for optimal pose detection:
        - Auto-rotate based on EXIF (handled before this)
        - Enhance contrast for low-light
        - Resize if too large (keep aspect ratio)
        """
        h, w = image.shape[:2]

        # Resize if too large (MediaPipe works best up to ~1920px)
        max_dim = 1920
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Auto-enhance contrast using CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return image

    def detect_pose(self, image: np.ndarray) -> Optional[PoseAngles]:
        """
        Detect pose landmarks and extract all relevant angles.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            PoseAngles dataclass with all extracted measurements,
            or None if detection fails.
        """
        processed = self.preprocess_image(image.copy())
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            logger.warning("No pose landmarks detected in image")
            return None

        landmarks = results.pose_landmarks.landmark
        h, w = processed.shape[:2]

        # Extract pixel coordinates and visibility for all landmarks
        lm_data = {}
        for name, idx in LANDMARKS_OF_INTEREST.items():
            lm = landmarks[idx]
            lm_data[name] = {
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z,
                "visibility": lm.visibility,
            }

        # Calculate average detection confidence for key points
        key_points = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                       "left_knee", "right_knee", "left_elbow", "right_elbow"]
        avg_conf = np.mean([lm_data[k]["visibility"] for k in key_points])

        pose_angles = PoseAngles()
        pose_angles.landmarks = lm_data
        pose_angles.detection_confidence = float(avg_conf)

        if avg_conf < self.min_confidence:
            pose_angles.warnings.append(
                f"Low detection confidence ({avg_conf:.2f}). Consider better lighting or camera angle."
            )

        # Determine which side is facing the camera (based on z-depth)
        left_z = np.mean([lm_data["left_shoulder"]["z"], lm_data["left_hip"]["z"]])
        right_z = np.mean([lm_data["right_shoulder"]["z"], lm_data["right_hip"]["z"]])
        pose_angles.detected_side = "left" if left_z < right_z else "right"

        # Validate side-view
        pose_angles.is_valid_side_view, pose_angles.side_view_confidence = self._validate_side_view(lm_data)
        if not pose_angles.is_valid_side_view:
            pose_angles.warnings.append(
                "Image may not be a proper side view. For best results, take photos from directly to the side."
            )

        # Extract angles using the side closest to camera
        side = pose_angles.detected_side
        opp = "right" if side == "left" else "left"

        # Use the visible side's landmarks
        shoulder = lm_data[f"{side}_shoulder"]
        elbow = lm_data[f"{side}_elbow"]
        wrist = lm_data[f"{side}_wrist"]
        hip = lm_data[f"{side}_hip"]
        knee = lm_data[f"{side}_knee"]
        ankle = lm_data[f"{side}_ankle"]
        ear = lm_data[f"{side}_ear"]

        # ── Knee Angle ──
        pose_angles.knee_angle = self._calculate_angle(
            (hip["x"], hip["y"]),
            (knee["x"], knee["y"]),
            (ankle["x"], ankle["y"])
        )

        # ── Elbow Angle ──
        pose_angles.elbow_angle = self._calculate_angle(
            (shoulder["x"], shoulder["y"]),
            (elbow["x"], elbow["y"]),
            (wrist["x"], wrist["y"])
        )

        # ── Trunk Recline Angle ──
        # Angle between vertical line through hip and hip→shoulder line
        # 90° = upright, >90° = reclined back, <90° = leaning forward
        hip_point = (hip["x"], hip["y"])
        shoulder_point = (shoulder["x"], shoulder["y"])
        vertical_point = (hip["x"], hip["y"] - 100)  # Point directly above hip

        trunk_angle = self._calculate_angle(vertical_point, hip_point, shoulder_point)
        # Convert to recline: 90° vertical → ~95-100° slight recline is ideal
        pose_angles.trunk_recline_angle = 90 + trunk_angle if shoulder["x"] > hip["x"] else 90 + trunk_angle
        # Simplified: measure the angle from vertical
        dx = shoulder["x"] - hip["x"]
        dy = hip["y"] - shoulder["y"]  # Inverted Y axis in images
        trunk_from_vertical = math.degrees(math.atan2(abs(dx), dy))
        pose_angles.trunk_recline_angle = 90 + trunk_from_vertical

        # ── Neck Flexion ──
        # Angle between ear→shoulder line and vertical
        ear_point = (ear["x"], ear["y"])
        sx, sy = shoulder["x"], shoulder["y"]
        ex, ey = ear["x"], ear["y"]
        neck_dx = ex - sx
        neck_dy = sy - ey  # Inverted Y
        neck_from_vertical = math.degrees(math.atan2(abs(neck_dx), neck_dy))
        # Positive = flexion (looking down), Negative = extension (looking up)
        pose_angles.neck_flexion = neck_from_vertical if ey > sy - (sy - ey) * 0.5 else -neck_from_vertical
        # Simplified: if ear is forward of shoulder, it's flexion
        if side == "right":
            pose_angles.neck_flexion = neck_from_vertical if ear["x"] < shoulder["x"] else -neck_from_vertical
        else:
            pose_angles.neck_flexion = neck_from_vertical if ear["x"] > shoulder["x"] else -neck_from_vertical

        # ── Wrist Extension ──
        # Approximate from wrist-elbow-hand angle deviation from straight
        # With just MediaPipe Pose (no hand landmarks), estimate from wrist position
        wrist_deviation = self._estimate_wrist_extension(elbow, wrist)
        pose_angles.wrist_extension = wrist_deviation

        # ── Shoulder Shrug Detection ──
        # Check if shoulders are elevated relative to normal position
        shoulder_ear_dist = abs(shoulder["y"] - ear["y"])
        head_height = abs(ear["y"] - lm_data["nose"]["y"])
        # If shoulder is very close to ear (<1.5x head height), likely shrugged
        pose_angles.shoulder_shrug = shoulder_ear_dist < (head_height * 2.0)

        # ── Feet on Floor ──
        # Estimate: if ankle is close to bottom of frame, likely on floor
        pose_angles.feet_on_floor = ankle["y"] > (h * 0.85)

        # ── Seat Pan Depth ──
        # Estimate space behind knee: ratio of knee-to-seat-edge
        # Approximate: if knee extends well past hip in X, seat may be too deep
        knee_hip_ratio = abs(knee["x"] - hip["x"]) / max(abs(shoulder["x"] - hip["x"]), 1)
        pose_angles.space_behind_knee_ok = 0.3 < knee_hip_ratio < 1.5

        return pose_angles

    def _validate_side_view(self, lm_data: Dict) -> Tuple[bool, float]:
        """
        Validate that the image is approximately a side view.
        Compare the horizontal distance between left/right shoulders
        vs the depth difference. In a true side view, one shoulder
        should be mostly occluded (small horizontal gap).
        """
        ls = lm_data["left_shoulder"]
        rs = lm_data["right_shoulder"]

        # Horizontal distance between shoulders (normalized)
        shoulder_width = abs(ls["x"] - rs["x"])

        # In a side view, shoulders should overlap significantly
        # Use hip distance as reference
        lh = lm_data["left_hip"]
        rh = lm_data["right_hip"]
        hip_width = abs(lh["x"] - rh["x"])

        # Average body landmark spread
        avg_spread = (shoulder_width + hip_width) / 2

        # Reference: shoulder-to-hip distance (should be larger than spread in side view)
        body_height = abs(ls["y"] - lh["y"])

        if body_height < 1:
            return False, 0.0

        ratio = avg_spread / body_height

        # In a perfect side view, ratio < 0.3; front view ~ 0.6+
        # Allow up to 0.5 for slightly angled views
        confidence = max(0.0, min(1.0, 1.0 - (ratio - 0.1) / 0.5))
        is_side = ratio < 0.5

        return is_side, confidence

    @staticmethod
    def _calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculate the angle at p2 formed by p1-p2-p3.
        Returns angle in degrees (0-180).
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return float(angle)

    @staticmethod
    def _estimate_wrist_extension(elbow: Dict, wrist: Dict) -> float:
        """
        Estimate wrist extension angle from elbow-wrist vector.
        Without detailed hand landmarks, we estimate based on
        the wrist position relative to elbow trajectory.
        Returns degrees of extension (positive = extended up).
        """
        # Simple estimation: angle of wrist relative to horizontal from elbow
        dx = wrist["x"] - elbow["x"]
        dy = elbow["y"] - wrist["y"]  # Inverted Y
        angle = math.degrees(math.atan2(dy, abs(dx) + 1e-8))
        return abs(angle)

    def draw_annotated_image(self, image: np.ndarray, pose_angles: PoseAngles) -> np.ndarray:
        """
        Draw an annotated image showing:
        - Skeleton with keypoints
        - Angle arcs with labels
        - Color-coded joint risk indicators
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        lm = pose_angles.landmarks
        side = pose_angles.detected_side

        # Color scheme
        COLOR_GOOD = (0, 200, 0)       # Green
        COLOR_WARN = (0, 165, 255)     # Orange
        COLOR_BAD = (0, 0, 255)        # Red
        COLOR_SKELETON = (255, 200, 0)  # Cyan-ish
        COLOR_TEXT = (255, 255, 255)    # White

        def get_point(name):
            return (int(lm[name]["x"]), int(lm[name]["y"]))

        def draw_angle_arc(center, p1, p2, angle, label, color):
            """Draw an angle arc and label at a joint."""
            cv2.line(annotated, center, p1, COLOR_SKELETON, 2)
            cv2.line(annotated, center, p2, COLOR_SKELETON, 2)

            # Draw arc
            radius = 30
            start_angle = math.degrees(math.atan2(p1[1] - center[1], p1[0] - center[0]))
            end_angle = math.degrees(math.atan2(p2[1] - center[1], p2[0] - center[0]))
            cv2.ellipse(annotated, center, (radius, radius), 0, start_angle, end_angle, color, 2)

            # Label
            label_pos = (center[0] + 35, center[1] - 10)
            cv2.putText(annotated, f"{label}: {angle:.0f}°", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw keypoints
        for name, data in lm.items():
            pt = (int(data["x"]), int(data["y"]))
            vis = data["visibility"]
            color = COLOR_GOOD if vis > 0.7 else COLOR_WARN if vis > 0.5 else COLOR_BAD
            cv2.circle(annotated, pt, 5, color, -1)
            cv2.circle(annotated, pt, 7, color, 1)

        # Draw skeleton connections
        connections = [
            (f"{side}_ear", f"{side}_shoulder"),
            (f"{side}_shoulder", f"{side}_elbow"),
            (f"{side}_elbow", f"{side}_wrist"),
            (f"{side}_shoulder", f"{side}_hip"),
            (f"{side}_hip", f"{side}_knee"),
            (f"{side}_knee", f"{side}_ankle"),
        ]

        for start, end in connections:
            if start in lm and end in lm:
                cv2.line(annotated, get_point(start), get_point(end), COLOR_SKELETON, 2)

        # Draw angle labels
        # Knee angle
        knee_color = COLOR_GOOD if 80 <= pose_angles.knee_angle <= 100 else COLOR_BAD
        knee_pt = get_point(f"{side}_knee")
        cv2.putText(annotated, f"Knee: {pose_angles.knee_angle:.0f}°",
                     (knee_pt[0] + 15, knee_pt[1]),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, knee_color, 2)

        # Elbow angle
        elbow_color = COLOR_GOOD if 80 <= pose_angles.elbow_angle <= 100 else COLOR_WARN
        elbow_pt = get_point(f"{side}_elbow")
        cv2.putText(annotated, f"Elbow: {pose_angles.elbow_angle:.0f}°",
                     (elbow_pt[0] + 15, elbow_pt[1]),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, elbow_color, 2)

        # Trunk recline
        trunk_color = COLOR_GOOD if 95 <= pose_angles.trunk_recline_angle <= 110 else COLOR_WARN
        hip_pt = get_point(f"{side}_hip")
        cv2.putText(annotated, f"Trunk: {pose_angles.trunk_recline_angle:.0f}°",
                     (hip_pt[0] + 15, hip_pt[1] - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, trunk_color, 2)

        # Neck flexion
        neck_color = COLOR_GOOD if abs(pose_angles.neck_flexion) <= 20 else COLOR_WARN
        shoulder_pt = get_point(f"{side}_shoulder")
        cv2.putText(annotated, f"Neck: {pose_angles.neck_flexion:.0f}°",
                     (shoulder_pt[0] + 15, shoulder_pt[1] - 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, neck_color, 2)

        # Detection confidence badge
        conf = pose_angles.detection_confidence
        conf_color = COLOR_GOOD if conf > 0.85 else COLOR_WARN if conf > 0.5 else COLOR_BAD
        cv2.putText(annotated, f"Confidence: {conf:.0%}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

        # Side view indicator
        sv_color = COLOR_GOOD if pose_angles.is_valid_side_view else COLOR_BAD
        sv_text = f"Side View: {'Yes' if pose_angles.is_valid_side_view else 'No'} ({pose_angles.side_view_confidence:.0%})"
        cv2.putText(annotated, sv_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sv_color, 2)

        # Detected side
        cv2.putText(annotated, f"Detected side: {pose_angles.detected_side}",
                     (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

        # Warnings
        y_offset = h - 20
        for warning in reversed(pose_angles.warnings):
            cv2.putText(annotated, f"! {warning}", (10, y_offset),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WARN, 1)
            y_offset -= 20

        return annotated

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
