"""
TactiVision CV — Field Mask Detection

Detects the playing field area using HSV color segmentation.
The pitch is green — we threshold for green pixels, find the largest
connected component (the field), and return a binary mask.

Usage:
    detector = FieldDetector()
    mask = detector.get_mask(frame)           # Binary mask (255 = field)
    is_on_field = detector.is_inside(x, y)    # Check if point is on field

How it works:
    1. Convert frame BGR → HSV
    2. Threshold for green hue (pitch color)
    3. Morphological operations to clean noise
    4. Find largest contour = the field
    5. Fill contour to get solid mask

Why HSV instead of RGB:
    HSV separates color (Hue) from brightness (Value).
    This means the green detection works under different lighting
    conditions — shadows, sun, floodlights. RGB would need different
    thresholds for each lighting condition.
"""

import cv2
import numpy as np


class FieldDetector:
    """
    Detects the football pitch area in a frame using green color segmentation.
    
    The mask is cached and only recomputed every N frames (the pitch doesn't
    move, so recomputing every frame is wasteful).
    """

    def __init__(
        self,
        hue_min: int = 30,
        hue_max: int = 80,
        sat_min: int = 30,
        val_min: int = 40,
        update_interval: int = 50,
        min_field_ratio: float = 0.15
    ):
        """
        Args:
            hue_min: Minimum hue for green detection (0-180 in OpenCV).
                     Default 30 covers yellow-green to green.
            hue_max: Maximum hue for green detection.
                     Default 80 covers green to cyan-green.
            sat_min: Minimum saturation. Filters out gray/white areas
                     that happen to have green-ish hue.
            val_min: Minimum value (brightness). Filters out very dark areas.
            update_interval: Recompute mask every N frames. The field doesn't
                            move, so we cache the mask between updates.
            min_field_ratio: Minimum fraction of frame that must be green to
                           consider it a valid field detection. If less than
                           this, something went wrong (e.g., close-up replay).
        """
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.sat_min = sat_min
        self.val_min = val_min
        self.update_interval = update_interval
        self.min_field_ratio = min_field_ratio

        # Cached state
        self._mask = None
        self._frame_count = 0
        self._last_valid_mask = None

    def get_mask(self, frame: np.ndarray, force_update: bool = False) -> np.ndarray:
        """
        Get the field mask for the given frame.

        Args:
            frame: BGR image (OpenCV format).
            force_update: If True, recompute even if cache is fresh.

        Returns:
            Binary mask where 255 = field, 0 = not field.
            Same height/width as input frame.
        """
        self._frame_count += 1

        # Use cached mask if still fresh
        if not force_update and self._mask is not None:
            if self._frame_count % self.update_interval != 0:
                return self._mask

        # Step 1: Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 2: Threshold for green
        # HSV ranges in OpenCV: H=0-180, S=0-255, V=0-255
        lower_green = np.array([self.hue_min, self.sat_min, self.val_min])
        upper_green = np.array([self.hue_max, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Step 3: Morphological operations to clean up
        # Close small holes inside the field (players, lines, etc.)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_close)

        # Remove small noise blobs outside the field (green ads, trees, etc.)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_open)

        # Step 4: Find the largest contour (the field)
        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # No green found — use last valid mask or return empty
            if self._last_valid_mask is not None:
                self._mask = self._last_valid_mask
            else:
                self._mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            return self._mask

        # Take the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Step 5: Fill the contour to get a solid mask
        field_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(field_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Validate: field should cover at least min_field_ratio of the frame
        field_ratio = np.count_nonzero(field_mask) / field_mask.size
        if field_ratio < self.min_field_ratio:
            # Probably a close-up or replay — use last valid mask
            if self._last_valid_mask is not None:
                self._mask = self._last_valid_mask
            else:
                self._mask = field_mask
        else:
            # Add some padding (dilate) to include players standing at the edge
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
            field_mask = cv2.dilate(field_mask, kernel_dilate)

            self._mask = field_mask
            self._last_valid_mask = field_mask

        return self._mask

    def is_inside(self, x: float, y: float, frame: np.ndarray = None) -> bool:
        """
        Check if a point (x, y) is inside the field.

        Args:
            x: X coordinate (pixel).
            y: Y coordinate (pixel).
            frame: If provided and no mask exists yet, compute mask first.

        Returns:
            True if the point is on the field, False otherwise.
        """
        if self._mask is None:
            if frame is not None:
                self.get_mask(frame)
            else:
                return True  # No mask available, assume inside

        # Bounds check
        h, w = self._mask.shape
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return False

        return self._mask[iy, ix] > 0

    def filter_detections(
        self,
        detections: list,
        frame: np.ndarray = None
    ) -> list:
        """
        Filter a list of detections, keeping only those inside the field.

        Args:
            detections: List of dicts with 'x' and 'y' keys (center point).
            frame: Current frame (used to update mask if needed).

        Returns:
            Filtered list containing only on-field detections.
        """
        if frame is not None:
            self.get_mask(frame)

        if self._mask is None:
            return detections

        return [d for d in detections if self.is_inside(d['x'], d['y'])]

    def visualize(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Overlay the field mask on the frame for debugging.

        Args:
            frame: BGR image.
            alpha: Transparency of the overlay (0 = invisible, 1 = opaque).

        Returns:
            Frame with green overlay on detected field area.
        """
        mask = self.get_mask(frame)
        overlay = frame.copy()

        # Color the field area green
        green_overlay = np.zeros_like(frame)
        green_overlay[:, :] = (0, 255, 0)  # BGR green

        # Apply only where mask is active
        mask_bool = mask > 0
        overlay[mask_bool] = cv2.addWeighted(
            frame[mask_bool], 1 - alpha,
            green_overlay[mask_bool], alpha,
            0
        )

        return overlay


# --- Standalone test ---
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m cv.field_mask <image_or_video>")
        print("  Displays the detected field mask overlay.")
        sys.exit(1)

    source = sys.argv[1]
    detector = FieldDetector()

    # Try as image first
    frame = cv2.imread(source)
    if frame is not None:
        result = detector.visualize(frame)
        cv2.imwrite('output/field_mask_test.jpg', result)
        mask = detector.get_mask(frame)
        field_pct = np.count_nonzero(mask) / mask.size * 100
        print(f"Field coverage: {field_pct:.1f}% of frame")
        print(f"Saved: output/field_mask_test.jpg")
    else:
        # Try as video — process first 5 frames
        import supervision as sv
        video_info = sv.VideoInfo.from_video_path(source)
        for idx, frame in enumerate(sv.get_video_frames_generator(source)):
            if idx >= 5:
                break
            mask = detector.get_mask(frame, force_update=True)
            field_pct = np.count_nonzero(mask) / mask.size * 100
            print(f"Frame {idx}: field = {field_pct:.1f}% of frame")

            if idx == 0:
                result = detector.visualize(frame)
                cv2.imwrite('output/field_mask_test.jpg', result)
                print(f"Saved: output/field_mask_test.jpg")
