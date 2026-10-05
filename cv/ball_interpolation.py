"""
TactiVision CV — Ball Interpolation

Problem:
    The ball is only detected in ~14% of frames (1005/7376).
    In the other 86%, the ball is still on screen but the model missed it.

Solution:
    Between two real detections, estimate the ball's position using
    interpolation. The ball travels in roughly straight lines (or gentle
    curves) over short time gaps.

How it works:
    1. Take the raw ball detections (frame, x, y)
    2. Remove static false positives (penalty spot)
    3. Find pairs of detections with a gap between them
    4. If the gap is small enough (< max_gap frames), interpolate
    5. Use linear interpolation for short gaps, quadratic for longer ones
    6. Mark interpolated positions with lower confidence

Limitations:
    - Can't interpolate across scene cuts (replays, camera switches)
    - Can't interpolate very long gaps (ball might change direction)
    - Doesn't handle bounces (ball changes trajectory on bounce)

Usage:
    from cv.ball_interpolation import BallInterpolator

    interpolator = BallInterpolator()
    filled = interpolator.interpolate(ball_detections)
    # filled has more entries than ball_detections
"""

import numpy as np
from typing import List, Dict, Optional


class BallInterpolator:
    """
    Fills gaps in ball detection by interpolating between known positions.

    The ball follows predictable trajectories over short time spans:
    - Passes: roughly linear
    - Aerial balls: parabolic (but we use linear for simplicity)
    - Stationary ball: constant position

    We only interpolate across gaps shorter than max_gap_frames.
    Longer gaps likely mean the ball went off-screen or there was a replay.
    """

    def __init__(
        self,
        max_gap_frames: int = 50,
        max_speed_px_per_frame: float = 80.0,
        static_threshold_px: float = 5.0,
        static_min_frames: int = 20,
        interpolation_confidence: float = 0.3
    ):
        """
        Args:
            max_gap_frames: Maximum gap (in frames) to interpolate across.
                           At 25fps, 50 frames = 2 seconds.
                           Beyond this, we don't trust the interpolation.

            max_speed_px_per_frame: Maximum reasonable ball speed in pixels/frame.
                                   If two detections imply speed > this, the ball
                                   probably jumped (scene cut) — don't interpolate.
                                   80px/frame at 25fps ≈ 2000px/sec = very fast shot.

            static_threshold_px: If a detection stays within this radius for
                                static_min_frames, it's a false positive (penalty spot).

            static_min_frames: Minimum consecutive similar-position frames to
                             consider a detection as static (false positive).

            interpolation_confidence: Confidence value assigned to interpolated
                                    positions. Lower than real detections so
                                    downstream code can distinguish them.
        """
        self.max_gap_frames = max_gap_frames
        self.max_speed_px_per_frame = max_speed_px_per_frame
        self.static_threshold_px = static_threshold_px
        self.static_min_frames = static_min_frames
        self.interpolation_confidence = interpolation_confidence

    def interpolate(self, ball_detections: List[Dict]) -> List[Dict]:
        """
        Main entry point. Takes raw ball detections, returns enriched list
        with interpolated positions filling the gaps.

        Args:
            ball_detections: List of dicts with keys:
                - 'frame': int (frame number)
                - 'x': float (ball center x)
                - 'y': float (ball center y)
                - 'confidence': float (detection confidence)

        Returns:
            List of dicts with same structure. Original detections are preserved
            (with 'interpolated': False). New interpolated entries have
            'interpolated': True and lower confidence.
        """
        if len(ball_detections) < 2:
            return ball_detections

        # Step 1: Sort by frame
        sorted_dets = sorted(ball_detections, key=lambda d: d['frame'])

        # Step 2: Remove static false positives (penalty spot, center dot)
        filtered_dets = self._remove_static_detections(sorted_dets)

        # Step 3: Interpolate gaps
        result = self._fill_gaps(filtered_dets)

        return result

    def _remove_static_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove detections that are stuck at the same position for many frames.
        These are almost certainly the penalty spot or center circle dot.

        Logic:
            Group consecutive detections by position.
            If a group stays within static_threshold_px for >= static_min_frames,
            remove the entire group.
        """
        if len(detections) < self.static_min_frames:
            return detections

        # Find clusters of static detections
        static_frames = set()
        i = 0

        while i < len(detections):
            # Start a potential static cluster
            cluster_start = i
            cluster_x = detections[i]['x']
            cluster_y = detections[i]['y']

            # Extend cluster while positions are within threshold
            j = i + 1
            while j < len(detections):
                dx = detections[j]['x'] - cluster_x
                dy = detections[j]['y'] - cluster_y
                dist = np.sqrt(dx**2 + dy**2)

                if dist <= self.static_threshold_px:
                    j += 1
                else:
                    break

            cluster_size = j - cluster_start

            # If cluster is large enough, mark as static
            if cluster_size >= self.static_min_frames:
                for k in range(cluster_start, j):
                    static_frames.add(detections[k]['frame'])

            i = j if j > i + 1 else i + 1

        # Filter out static detections
        filtered = [d for d in detections if d['frame'] not in static_frames]

        removed = len(detections) - len(filtered)
        if removed > 0:
            print(f"  Static filter: removed {removed} false positives "
                  f"(penalty spot / center dot)")

        return filtered

    def _fill_gaps(self, detections: List[Dict]) -> List[Dict]:
        """
        Interpolate between consecutive detections where the gap is small enough.

        For each pair of consecutive real detections (A at frame f1, B at frame f2):
            - If f2 - f1 <= max_gap_frames: fill frames f1+1 to f2-1
            - If speed between A and B is reasonable (< max_speed): interpolate
            - Otherwise: skip (likely a scene cut or the ball changed direction)
        """
        result = []
        interpolated_count = 0

        for i in range(len(detections)):
            # Add the real detection
            entry = detections[i].copy()
            entry['interpolated'] = False
            result.append(entry)

            # Check if there's a next detection to interpolate toward
            if i >= len(detections) - 1:
                continue

            current = detections[i]
            next_det = detections[i + 1]

            # Calculate gap
            frame_gap = next_det['frame'] - current['frame']

            # Skip if no gap (consecutive frames already detected)
            if frame_gap <= 1:
                continue

            # Skip if gap is too large (can't reliably interpolate)
            if frame_gap > self.max_gap_frames:
                continue

            # Check speed reasonableness
            dx = next_det['x'] - current['x']
            dy = next_det['y'] - current['y']
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance / frame_gap  # pixels per frame

            if speed > self.max_speed_px_per_frame:
                # Ball teleported — scene cut or error, don't interpolate
                continue

            # Interpolate: linear between current and next
            for f in range(current['frame'] + 1, next_det['frame']):
                # t goes from 0 (at current) to 1 (at next)
                t = (f - current['frame']) / frame_gap

                interp_x = current['x'] + t * dx
                interp_y = current['y'] + t * dy

                # Confidence decreases toward the middle of the gap
                # (most uncertain in the middle, most certain near endpoints)
                distance_from_nearest = min(t, 1 - t)  # 0 at edges, 0.5 at center
                conf = self.interpolation_confidence * (1 - distance_from_nearest)

                result.append({
                    'frame': f,
                    'x': float(interp_x),
                    'y': float(interp_y),
                    'confidence': float(conf),
                    'interpolated': True
                })
                interpolated_count += 1

        # Sort by frame (interpolated entries are out of order)
        result.sort(key=lambda d: d['frame'])

        # Summary
        real_count = sum(1 for d in result if not d['interpolated'])
        print(f"  Interpolation: {real_count} real + {interpolated_count} interpolated "
              f"= {len(result)} total")

        return result

    def get_coverage_stats(
        self,
        ball_data: List[Dict],
        total_frames: int
    ) -> Dict:
        """
        Compute coverage statistics.

        Args:
            ball_data: Ball positions (after interpolation).
            total_frames: Total frames in video.

        Returns:
            Dict with coverage metrics.
        """
        frames_with_ball = set(d['frame'] for d in ball_data)
        real_frames = set(d['frame'] for d in ball_data if not d.get('interpolated', False))
        interp_frames = frames_with_ball - real_frames

        return {
            'total_frames': total_frames,
            'real_detections': len(real_frames),
            'interpolated_frames': len(interp_frames),
            'total_coverage': len(frames_with_ball),
            'real_coverage_pct': len(real_frames) / total_frames * 100,
            'total_coverage_pct': len(frames_with_ball) / total_frames * 100,
            'improvement_factor': len(frames_with_ball) / max(len(real_frames), 1)
        }


# --- Standalone test ---
if __name__ == '__main__':
    import json
    import sys

    # Load ball detections
    ball_path = 'output/tracked_v5_ball.json'
    if len(sys.argv) > 1:
        ball_path = sys.argv[1]

    with open(ball_path) as f:
        ball_data = json.load(f)

    print(f"Input: {len(ball_data)} ball detections")
    print(f"Frames covered: {len(set(d['frame'] for d in ball_data))}")

    # Run interpolation
    interpolator = BallInterpolator()
    result = interpolator.interpolate(ball_data)

    # Stats
    total_frames = max(d['frame'] for d in ball_data) + 1
    stats = interpolator.get_coverage_stats(result, total_frames)

    print(f"\n=== Ball Coverage ===")
    print(f"  Before: {stats['real_detections']}/{stats['total_frames']} "
          f"({stats['real_coverage_pct']:.1f}%)")
    print(f"  After:  {stats['total_coverage']}/{stats['total_frames']} "
          f"({stats['total_coverage_pct']:.1f}%)")
    print(f"  Improvement: {stats['improvement_factor']:.1f}x")

    # Save
    output_path = ball_path.replace('.json', '_interpolated.json')
    with open(output_path, 'w') as f:
        json.dump(result, f)
    print(f"\n  Saved: {output_path}")
