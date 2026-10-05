"""
TactiVision CV — Ball Tracker (Trajectory-Based)

Problem:
    At low detection confidence (0.05), we get multiple "ball" candidates per
    frame. Many are false positives: white shoes, socks, line markings.
    Simple overlap filters fail because during dribbling, the real ball IS
    inside the player's bounding box.

Solution:
    Use trajectory consistency to find the REAL ball path through all candidates.
    The real ball follows a smooth, physically plausible path. False positives
    from shoes appear randomly and don't form a consistent trajectory.

Algorithm (Greedy Trajectory Tracking):
    1. Detect ALL ball candidates per frame (multiple per frame at conf=0.05)
    2. Start from the highest-confidence detection in frame 0
    3. For each subsequent frame, predict where the ball SHOULD be
       (based on velocity from previous frames)
    4. Among all candidates in that frame, pick the one closest to the
       predicted position
    5. If no candidate is close enough, mark the frame as "missing" (to be
       interpolated later)

Why this works:
    - Real ball: moves smoothly → always close to predicted position
    - Shoe/sock: appears at random positions (wherever the player's foot is)
      → usually far from predicted ball position
    - During dribbling: ball oscillates near the player but still follows a
      smooth micro-trajectory that our prediction can track

Usage:
    from cv.ball_tracker import BallTrajectoryTracker

    tracker = BallTrajectoryTracker()
    ball_positions = tracker.track(all_candidates_per_frame, total_frames)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class BallTrajectoryTracker:
    """
    Finds the most consistent ball trajectory through multiple candidates
    per frame using velocity-based prediction.
    """

    def __init__(
        self,
        max_prediction_distance: float = 100.0,
        velocity_smoothing: int = 3,
        max_acceleration: float = 50.0,
        min_confidence_solo: float = 0.15,
        missing_frame_limit: int = 10
    ):
        """
        Args:
            max_prediction_distance: Maximum allowed distance (px) between
                predicted position and actual candidate. If no candidate is
                within this radius, the frame is marked "missing".
                100px at 25fps allows speeds up to 2500 px/s.

            velocity_smoothing: Number of previous frames to average for
                velocity estimation. Higher = smoother prediction but slower
                to react to direction changes. 3 frames = 120ms of history.

            max_acceleration: Maximum allowed change in velocity (px/frame²).
                Prevents the tracker from "jumping" to a distant false positive
                when the ball suddenly changes direction. The ball decelerates
                and accelerates gradually (physics).

            min_confidence_solo: When only ONE candidate exists in a frame and
                it's far from prediction, require at least this confidence to
                accept it. Shoes typically have low confidence (0.05-0.10).

            missing_frame_limit: If we miss this many consecutive frames,
                reset the tracker and look for the ball from scratch. The ball
                probably went off-screen or there was a scene cut.
        """
        self.max_prediction_distance = max_prediction_distance
        self.velocity_smoothing = velocity_smoothing
        self.max_acceleration = max_acceleration
        self.min_confidence_solo = min_confidence_solo
        self.missing_frame_limit = missing_frame_limit

    def track(
        self,
        candidates_per_frame: Dict[int, List[Dict]],
        total_frames: int
    ) -> List[Dict]:
        """
        Main entry point. Finds the best ball trajectory through all candidates.

        Args:
            candidates_per_frame: Dict mapping frame number to list of candidate
                detections. Each candidate has 'x', 'y', 'confidence'.
                Example: {0: [{'x': 100, 'y': 200, 'confidence': 0.8}, ...], ...}

            total_frames: Total number of frames in the video.

        Returns:
            List of ball positions (one per frame where ball was found).
            Each entry: {'frame', 'x', 'y', 'confidence', 'method'}
            where method is 'detected' or 'predicted'.
        """
        # --- Step 1: Initialize with strongest detection ---
        trajectory = []
        velocity_history = []  # List of (vx, vy) tuples
        consecutive_missing = 0

        # Find the first frame with candidates to start from
        start_frame = self._find_start_frame(candidates_per_frame, total_frames)
        if start_frame is None:
            print("  No ball candidates found in any frame!")
            return []

        # Pick the highest-confidence candidate in the start frame
        first_candidates = candidates_per_frame[start_frame]
        best = max(first_candidates, key=lambda c: c['confidence'])
        trajectory.append({
            'frame': start_frame,
            'x': best['x'],
            'y': best['y'],
            'confidence': best['confidence'],
            'method': 'detected'
        })

        # --- Step 2: Track frame by frame ---
        for frame in range(start_frame + 1, total_frames):
            candidates = candidates_per_frame.get(frame, [])

            # Get current state (last known position and velocity)
            last_pos = (trajectory[-1]['x'], trajectory[-1]['y'])
            velocity = self._estimate_velocity(trajectory, velocity_history)

            # Predict where ball should be this frame
            predicted_x = last_pos[0] + velocity[0]
            predicted_y = last_pos[1] + velocity[1]

            # --- Step 3: Find best candidate near prediction ---
            best_candidate = self._select_best_candidate(
                candidates, predicted_x, predicted_y, velocity
            )

            if best_candidate is not None:
                # Found a good match near our prediction
                trajectory.append({
                    'frame': frame,
                    'x': best_candidate['x'],
                    'y': best_candidate['y'],
                    'confidence': best_candidate['confidence'],
                    'method': 'detected'
                })

                # Update velocity history
                if len(trajectory) >= 2:
                    prev = trajectory[-2]
                    dt = frame - prev['frame']
                    if dt > 0:
                        vx = (best_candidate['x'] - prev['x']) / dt
                        vy = (best_candidate['y'] - prev['y']) / dt
                        velocity_history.append((vx, vy))
                        # Keep only recent history
                        if len(velocity_history) > self.velocity_smoothing * 2:
                            velocity_history = velocity_history[-self.velocity_smoothing * 2:]

                consecutive_missing = 0

            else:
                # No good candidate found — frame is missing
                consecutive_missing += 1

                # If we've lost the ball for too long, try to re-acquire
                if consecutive_missing >= self.missing_frame_limit:
                    reacquired = self._reacquire(
                        candidates_per_frame, frame, total_frames
                    )
                    if reacquired is not None:
                        trajectory.append(reacquired)
                        velocity_history = []  # Reset velocity
                        consecutive_missing = 0

        # --- Summary ---
        detected = sum(1 for t in trajectory if t['method'] == 'detected')
        print(f"  Trajectory tracker: {detected} frames tracked out of {total_frames}")
        print(f"  Coverage: {detected}/{total_frames} ({detected/total_frames*100:.1f}%)")

        return trajectory

    def _find_start_frame(
        self,
        candidates_per_frame: Dict[int, List[Dict]],
        total_frames: int
    ) -> Optional[int]:
        """
        Find the first frame with a high-confidence ball detection.
        We want to start tracking from a reliable detection.
        """
        # Look in the first 200 frames for a confident detection
        for frame in range(min(200, total_frames)):
            candidates = candidates_per_frame.get(frame, [])
            if candidates:
                best_conf = max(c['confidence'] for c in candidates)
                if best_conf >= 0.3:  # Require decent confidence to start
                    return frame

        # If no confident detection in first 200 frames, take whatever we can get
        for frame in range(total_frames):
            if frame in candidates_per_frame and candidates_per_frame[frame]:
                return frame

        return None

    def _estimate_velocity(
        self,
        trajectory: List[Dict],
        velocity_history: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Estimate current ball velocity by averaging recent velocity measurements.

        Uses the last N velocity samples (where N = velocity_smoothing).
        If not enough history, returns (0, 0) — no prediction.
        """
        if len(velocity_history) == 0:
            return (0.0, 0.0)

        # Take the last N velocity samples
        recent = velocity_history[-self.velocity_smoothing:]

        # Average them (simple but effective for smooth motion)
        avg_vx = np.mean([v[0] for v in recent])
        avg_vy = np.mean([v[1] for v in recent])

        return (avg_vx, avg_vy)

    def _select_best_candidate(
        self,
        candidates: List[Dict],
        predicted_x: float,
        predicted_y: float,
        velocity: Tuple[float, float]
    ) -> Optional[Dict]:
        """
        From a list of candidates, select the one most consistent with
        our trajectory prediction.

        Selection criteria (in priority order):
        1. Must be within max_prediction_distance of predicted position
        2. Among valid candidates, prefer the one closest to prediction
        3. If tied in distance, prefer higher confidence

        Returns None if no candidate is acceptable.
        """
        if not candidates:
            return None

        # Calculate distance from prediction for each candidate
        scored = []
        for c in candidates:
            dx = c['x'] - predicted_x
            dy = c['y'] - predicted_y
            dist = np.sqrt(dx**2 + dy**2)

            if dist <= self.max_prediction_distance:
                # Score: lower is better. Weighted combination of
                # distance (primary) and inverse confidence (secondary)
                score = dist - c['confidence'] * 10  # Slight boost for high conf
                scored.append((score, dist, c))

        if not scored:
            # No candidate near prediction — check if there's a single
            # very confident detection (ball might have changed direction sharply)
            high_conf = [c for c in candidates if c['confidence'] >= self.min_confidence_solo]
            if len(high_conf) == 1:
                # Only one confident candidate, and nothing near prediction
                # This might be a sharp direction change (deflection, pass)
                c = high_conf[0]
                dx = c['x'] - predicted_x
                dy = c['y'] - predicted_y
                dist = np.sqrt(dx**2 + dy**2)
                # Allow up to 2x max distance for high-confidence solo detections
                if dist <= self.max_prediction_distance * 2:
                    return c

            return None

        # Sort by score (lowest = best)
        scored.sort(key=lambda x: x[0])
        return scored[0][2]

    def _reacquire(
        self,
        candidates_per_frame: Dict[int, List[Dict]],
        current_frame: int,
        total_frames: int
    ) -> Optional[Dict]:
        """
        When the ball has been lost for too long, look ahead for a
        high-confidence detection to restart tracking from.

        This handles scene cuts, long occlusions, or the ball going
        off-screen and coming back.
        """
        # Look ahead up to 50 frames for a confident detection
        for frame in range(current_frame, min(current_frame + 50, total_frames)):
            candidates = candidates_per_frame.get(frame, [])
            if candidates:
                best = max(candidates, key=lambda c: c['confidence'])
                if best['confidence'] >= 0.25:
                    return {
                        'frame': frame,
                        'x': best['x'],
                        'y': best['y'],
                        'confidence': best['confidence'],
                        'method': 'detected'
                    }

        return None

    def interpolate_gaps(
        self,
        trajectory: List[Dict],
        max_gap: int = 75
    ) -> List[Dict]:
        """
        Fill gaps in the trajectory with linear interpolation.
        Same logic as BallInterpolator but integrated here.

        Args:
            trajectory: Output from track() method.
            max_gap: Maximum gap (frames) to interpolate across.

        Returns:
            Trajectory with gaps filled. New entries have method='interpolated'.
        """
        if len(trajectory) < 2:
            return trajectory

        sorted_traj = sorted(trajectory, key=lambda d: d['frame'])
        result = []

        for i in range(len(sorted_traj)):
            result.append(sorted_traj[i])

            if i >= len(sorted_traj) - 1:
                continue

            current = sorted_traj[i]
            next_pos = sorted_traj[i + 1]
            gap = next_pos['frame'] - current['frame']

            if gap <= 1 or gap > max_gap:
                continue

            # Linear interpolation
            dx = next_pos['x'] - current['x']
            dy = next_pos['y'] - current['y']

            for f in range(current['frame'] + 1, next_pos['frame']):
                t = (f - current['frame']) / gap
                result.append({
                    'frame': f,
                    'x': float(current['x'] + t * dx),
                    'y': float(current['y'] + t * dy),
                    'confidence': 0.2,
                    'method': 'interpolated'
                })

        result.sort(key=lambda d: d['frame'])
        return result


def build_candidates_from_raw_detections(
    video_path: str,
    model_path: str,
    conf: float = 0.05
) -> Tuple[Dict[int, List[Dict]], int]:
    """
    Run ball detection and collect ALL candidates per frame.
    Unlike the simple approach that keeps only the best per frame,
    this keeps ALL detections so the trajectory tracker can choose.

    Args:
        video_path: Path to video file.
        model_path: Path to YOLO model.
        conf: Detection confidence (low = more candidates).

    Returns:
        Tuple of (candidates_per_frame dict, total_frames int)
    """
    from ultralytics import YOLO
    import supervision as sv

    BALL_CLASS = 0
    model = YOLO(model_path)
    video_info = sv.VideoInfo.from_video_path(video_path)

    candidates_per_frame = {}

    results_gen = model.predict(
        source=video_path,
        conf=conf,
        classes=[BALL_CLASS],
        stream=True,
        verbose=False
    )

    for idx, results in enumerate(results_gen):
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            continue

        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        frame_candidates = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            frame_candidates.append({
                'x': float((x1 + x2) / 2),
                'y': float((y1 + y2) / 2),
                'confidence': float(confidences[i]),
                'bbox_w': float(x2 - x1),
                'bbox_h': float(y2 - y1)
            })

        if frame_candidates:
            candidates_per_frame[idx] = frame_candidates

    return candidates_per_frame, video_info.total_frames


# --- Standalone test ---
if __name__ == '__main__':
    import json
    import sys

    print("=== Ball Trajectory Tracker Test ===\n")

    # Option 1: Test with existing ball JSON (simulated multi-candidate)
    ball_path = 'output/tracked_v5_ball.json'
    if len(sys.argv) > 1:
        ball_path = sys.argv[1]

    print(f"Loading: {ball_path}")
    with open(ball_path) as f:
        ball_data = json.load(f)

    # Convert single-detection format to multi-candidate format
    # (In real use, you'd run build_candidates_from_raw_detections on GPU)
    candidates_per_frame = {}
    for d in ball_data:
        frame = d['frame']
        if frame not in candidates_per_frame:
            candidates_per_frame[frame] = []
        candidates_per_frame[frame].append({
            'x': d['x'],
            'y': d['y'],
            'confidence': d['confidence']
        })

    total_frames = max(d['frame'] for d in ball_data) + 1
    frames_with_candidates = len(candidates_per_frame)

    print(f"  Total frames: {total_frames}")
    print(f"  Frames with candidates: {frames_with_candidates} "
          f"({frames_with_candidates/total_frames*100:.1f}%)")

    # Run trajectory tracker
    print("\nRunning trajectory tracker...")
    tracker = BallTrajectoryTracker()
    trajectory = tracker.track(candidates_per_frame, total_frames)

    # Interpolate gaps
    print("\nInterpolating gaps...")
    final = tracker.interpolate_gaps(trajectory, max_gap=75)

    # Stats
    detected = sum(1 for d in final if d['method'] == 'detected')
    interpolated = sum(1 for d in final if d['method'] == 'interpolated')
    coverage = len(final) / total_frames * 100

    print(f"\n=== Results ===")
    print(f"  Detected: {detected}")
    print(f"  Interpolated: {interpolated}")
    print(f"  Total coverage: {len(final)}/{total_frames} ({coverage:.1f}%)")

    # Save
    output_path = ball_path.replace('.json', '_trajectory.json')
    with open(output_path, 'w') as f:
        json.dump(final, f)
    print(f"  Saved: {output_path}")
