"""
TactiVision CV — Team Classification
Assigns players to Home/Away/Referee using jersey color (KMeans on HSV).
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self):
        self.kmeans = None
        self.team_labels = ['team_a', 'team_b', 'referee']
        self.calibrated = False

    def get_jersey_color(self, frame, bbox):
        """Extract dominant HSV color from player's torso region."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0, 0, 0])

        h, w = crop.shape[:2]
        # Torso = top 15% to 45% of bounding box
        torso = crop[int(h * 0.15):int(h * 0.45), :]
        if torso.size == 0:
            return np.array([0, 0, 0])

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        return np.median(hsv.reshape(-1, 3), axis=0)

    def calibrate(self, frame, bboxes):
        """Calibrate team colors from a set of player bounding boxes."""
        colors = []
        for bbox in bboxes:
            color = self.get_jersey_color(frame, bbox)
            colors.append(color)

        if len(colors) < 6:
            return False

        colors = np.array(colors)
        self.kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        self.kmeans.fit(colors)
        self.calibrated = True
        return True

    def classify(self, frame, bbox):
        """Classify a single player into team_a/team_b/referee."""
        if not self.calibrated:
            return 'unknown'

        color = self.get_jersey_color(frame, bbox)
        label = self.kmeans.predict([color])[0]
        return self.team_labels[label]
