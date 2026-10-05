"""
TactiVision CV — Player Tracker v5 (optimized)
Single model pass, BoT-SORT tracking, post-merge.
Designed to run in ~30-40 min on CPU.
"""

import supervision as sv
from ultralytics import YOLO
import numpy as np
import cv2
import json
from collections import Counter, defaultdict


# Model class mapping
BALL_CLASS = 0
GOALKEEPER_CLASS = 1
PLAYER_CLASS = 2
REFEREE_CLASS = 3

TRACK_CLASSES = [PLAYER_CLASS, GOALKEEPER_CLASS]


class PlayerTracker:
    def __init__(self, model_path='models/best.pt', conf=0.15, device=None):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def track_video(self, video_path, output_path='output/tracked_v5.mp4'):
        video_path = str(video_path)
        output_path = str(output_path)

        video_info = sv.VideoInfo.from_video_path(video_path)
        fps = video_info.fps or 25

        # === PASS 1: Single-pass tracking (players) + ball detection ===
        print("=== PASS 1: BoT-SORT Tracking ===")
        raw_tracking, ball_positions, track_colors = self._single_pass_track(video_path, video_info, fps)

        # === PASS 2: Merge fragments ===
        print("\n=== PASS 2: Merging Fragments ===")
        tracking_data = self._filter_short_tracks(raw_tracking, min_frames=int(fps))
        tracking_data = self._merge_by_proximity(tracking_data, max_gap=int(fps * 8), max_distance=250)
        tracking_data = self._merge_by_appearance(tracking_data, track_colors, max_gap=int(fps * 20))

        # === PASS 3: Write annotated video ===
        print("\n=== PASS 3: Writing Output Video ===")
        self._write_annotated_video(video_path, output_path, tracking_data, ball_positions, video_info)

        self._save_results(tracking_data, ball_positions, output_path, video_info.total_frames)
        return tracking_data, ball_positions

    def _single_pass_track(self, video_path, video_info, fps):
        """
        Single model.track() call — handles both players and ball.
        BoT-SORT tracks players, ball is extracted from same results (no second pass).
        """
        tracking_data = []
        ball_positions = []
        track_colors = defaultdict(list)

        import os
        tracker_cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfg', 'botsort_football.yaml')
        if not os.path.exists(tracker_cfg):
            tracker_cfg = 'botsort.yaml'

        # Single pass: detect ALL classes, track will only assign IDs to detected objects
        track_options = {
            'source': video_path,
            'conf': self.conf,
            'iou': 0.5,
            'tracker': tracker_cfg,
            'stream': True,
            'persist': True,
            'verbose': False,
        }
        if self.device:
            track_options['device'] = self.device

        results_gen = self.model.track(
            **track_options
        )

        for idx, results in enumerate(results_gen):
            frame = results.orig_img
            boxes = results.boxes

            if boxes is None or len(boxes) == 0:
                continue

            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id

            # --- Ball: extract from same results (no tracking needed) ---
            ball_mask = class_ids == BALL_CLASS
            if ball_mask.any():
                ball_boxes = xyxy[ball_mask]
                ball_confs = confidences[ball_mask]
                best = ball_confs.argmax()
                bx1, by1, bx2, by2 = ball_boxes[best]
                ball_positions.append({
                    "frame": idx,
                    "x": float((bx1 + bx2) / 2),
                    "y": float((by1 + by2) / 2),
                    "confidence": float(ball_confs[best])
                })

            # --- Players: use tracked IDs ---
            if track_ids is None:
                continue

            track_ids_np = track_ids.cpu().numpy().astype(int)
            player_mask = np.isin(class_ids, TRACK_CLASSES)

            for i in np.where(player_mask)[0]:
                x1, y1, x2, y2 = xyxy[i]
                tid = int(track_ids_np[i])
                cls = int(class_ids[i])

                tracking_data.append({
                    "frame": idx,
                    "track_id": tid,
                    "class_id": cls,
                    "class_name": "goalkeeper" if cls == GOALKEEPER_CLASS else "player",
                    "x": float((x1 + x2) / 2),
                    "y": float((y1 + y2) / 2),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(confidences[i])
                })

                # Sample jersey color every 10 frames (save CPU)
                if idx % 10 == 0 and len(track_colors[tid]) < 15:
                    color = self._extract_jersey_color(frame, [x1, y1, x2, y2])
                    if color is not None:
                        track_colors[tid].append(color)

            if idx % 100 == 0:
                n_players = int(player_mask.sum())
                unique_so_far = len(set(d['track_id'] for d in tracking_data))
                ball_str = "Yes" if ball_mask.any() else "No"
                print(f"  Frame {idx}/{video_info.total_frames} | "
                      f"Players: {n_players} | IDs: {unique_so_far} | Ball: {ball_str}")

        unique_total = len(set(d['track_id'] for d in tracking_data))
        print(f"  Done: {unique_total} unique IDs, {len(ball_positions)} ball detections")
        return tracking_data, ball_positions, track_colors

    def _extract_jersey_color(self, frame, bbox):
        """Extract median HSV from player torso."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 5:
            return None

        h, w = crop.shape[:2]
        torso = crop[int(h * 0.15):int(h * 0.45), :]
        if torso.size == 0:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        return np.median(hsv.reshape(-1, 3), axis=0)

    def _filter_short_tracks(self, tracking_data, min_frames=25):
        """Remove tracks shorter than min_frames."""
        id_counts = Counter(d['track_id'] for d in tracking_data)
        valid_ids = {tid for tid, count in id_counts.items() if count >= min_frames}
        filtered = [d for d in tracking_data if d['track_id'] in valid_ids]
        print(f"  Filter: {len(id_counts)} → {len(valid_ids)} tracks (removed < {min_frames} frames)")
        return filtered

    def _merge_by_proximity(self, tracking_data, max_gap=100, max_distance=150):
        """Merge tracks that end/start near each other."""
        if not tracking_data:
            return tracking_data

        track_timelines = defaultdict(list)
        for d in tracking_data:
            track_timelines[d['track_id']].append((d['frame'], d['x'], d['y']))
        for tid in track_timelines:
            track_timelines[tid].sort()

        track_info = {}
        for tid, tl in track_timelines.items():
            track_info[tid] = {
                'start_frame': tl[0][0], 'end_frame': tl[-1][0],
                'start_pos': (tl[0][1], tl[0][2]), 'end_pos': (tl[-1][1], tl[-1][2]),
                'length': len(tl)
            }

        sorted_tracks = sorted(track_info.keys(), key=lambda t: -track_info[t]['length'])
        merge_map = {}

        for i, anchor in enumerate(sorted_tracks):
            if anchor in merge_map:
                continue
            anchor_end = track_info[anchor]['end_frame']
            anchor_end_pos = track_info[anchor]['end_pos']

            for j in range(i + 1, len(sorted_tracks)):
                candidate = sorted_tracks[j]
                if candidate in merge_map:
                    continue
                c = track_info[candidate]

                gap = c['start_frame'] - anchor_end
                if gap < 1 or gap > max_gap:
                    continue

                dx = c['start_pos'][0] - anchor_end_pos[0]
                dy = c['start_pos'][1] - anchor_end_pos[1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist < max_distance:
                    merge_map[candidate] = anchor
                    if c['end_frame'] > anchor_end:
                        anchor_end = c['end_frame']
                        anchor_end_pos = c['end_pos']

        merged = [dict(d, track_id=merge_map.get(d['track_id'], d['track_id'])) for d in tracking_data]
        final = len(set(d['track_id'] for d in merged))
        print(f"  Proximity merge: {len(merge_map)} merged → {final} unique IDs")
        return merged

    def _merge_by_appearance(self, tracking_data, track_colors, max_gap=250):
        """Merge tracks with similar jersey color that don't overlap in time."""
        if not tracking_data or not track_colors:
            return tracking_data

        active_ids = set(d['track_id'] for d in tracking_data)
        track_avg_colors = {}
        for tid, colors in track_colors.items():
            if tid in active_ids and len(colors) >= 2:
                track_avg_colors[tid] = np.median(colors, axis=0)

        if len(track_avg_colors) < 2:
            return tracking_data

        track_frames = defaultdict(list)
        for d in tracking_data:
            track_frames[d['track_id']].append(d['frame'])
        track_spans = {tid: (min(f), max(f)) for tid, f in track_frames.items()}

        sorted_ids = sorted(track_avg_colors.keys(), key=lambda t: -(track_spans[t][1] - track_spans[t][0]))
        merge_map = {}

        for i, anchor in enumerate(sorted_ids):
            if anchor in merge_map:
                continue
            anchor_color = track_avg_colors[anchor]
            anchor_start, anchor_end = track_spans[anchor]

            for j in range(i + 1, len(sorted_ids)):
                candidate = sorted_ids[j]
                if candidate in merge_map:
                    continue
                cand_color = track_avg_colors[candidate]
                cand_start, cand_end = track_spans[candidate]

                # No temporal overlap
                if cand_start <= anchor_end and cand_end >= anchor_start:
                    continue
                gap = min(abs(cand_start - anchor_end), abs(anchor_start - cand_end))
                if gap > max_gap:
                    continue

                color_dist = np.linalg.norm(anchor_color - cand_color)
                if color_dist < 55.0:
                    merge_map[candidate] = anchor
                    if cand_end > anchor_end:
                        anchor_end = cand_end

        merged = [dict(d, track_id=merge_map.get(d['track_id'], d['track_id'])) for d in tracking_data]
        final = len(set(d['track_id'] for d in merged))
        print(f"  Appearance merge: {len(merge_map)} merged → {final} unique IDs")
        return merged

    def _write_annotated_video(self, video_path, output_path, tracking_data, ball_positions, video_info):
        """Write output video with final merged IDs."""
        frame_players = defaultdict(list)
        for d in tracking_data:
            frame_players[d['frame']].append(d)
        frame_balls = {b['frame']: b for b in ball_positions}

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5)

        with sv.VideoSink(output_path, video_info) as sink:
            for idx, frame in enumerate(sv.get_video_frames_generator(video_path)):
                players = frame_players.get(idx, [])
                if players:
                    xyxy = np.array([p['bbox'] for p in players])
                    track_ids = np.array([p['track_id'] for p in players])
                    confs = np.array([p.get('confidence', 0.5) for p in players])
                    class_ids = np.array([p.get('class_id', PLAYER_CLASS) for p in players])
                    dets = sv.Detections(xyxy=xyxy, confidence=confs, tracker_id=track_ids, class_id=class_ids)
                    labels = [f"#{tid}" for tid in track_ids]
                    frame = box_annotator.annotate(frame, dets)
                    frame = label_annotator.annotate(frame, dets, labels)

                if idx in frame_balls:
                    bx, by = int(frame_balls[idx]['x']), int(frame_balls[idx]['y'])
                    cv2.circle(frame, (bx, by), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (bx, by), 10, (0, 0, 0), 2)

                sink.write_frame(frame)
                if idx % 500 == 0:
                    print(f"  Writing frame {idx}/{video_info.total_frames}")

    def _save_results(self, tracking_data, ball_positions, output_path, total_frames):
        """Save to JSON."""
        players_path = output_path.replace('.mp4', '_players.json')
        ball_path = output_path.replace('.mp4', '_ball.json')
        with open(players_path, 'w') as f:
            json.dump(tracking_data, f)
        with open(ball_path, 'w') as f:
            json.dump(ball_positions, f)

        unique_ids = len(set(d['track_id'] for d in tracking_data))
        ball_coverage = len(ball_positions) / max(total_frames, 1) * 100
        print(f"\n=== FINAL RESULTS ===")
        print(f"  Frames: {total_frames}")
        print(f"  Unique player IDs: {unique_ids}")
        print(f"  Ball: {len(ball_positions)}/{total_frames} ({ball_coverage:.1f}%)")
        print(f"  Output: {players_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/match_5min.mp4')
    parser.add_argument('--output', default='output/tracked_v5.mp4')
    parser.add_argument('--model', default='models/best.pt')
    parser.add_argument('--conf', type=float, default=0.15)
    args = parser.parse_args()

    tracker = PlayerTracker(model_path=args.model, conf=args.conf)
    tracker.track_video(args.video, args.output)
