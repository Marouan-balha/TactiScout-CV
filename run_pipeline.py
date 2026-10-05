import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return f'sha256:{digest.hexdigest()}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the TactiScout CV observation worker.')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--device', default=None, help='Inference device, for example cpu or cuda:0')
    parser.add_argument('--sample-frames', type=int, default=25)
    parser.add_argument('--calibration-frame', type=int, default=None)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'{label} does not exist: {path}')


def write_json(path: Path, value: object) -> None:
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('w', encoding='utf-8') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()
    require_file(args.video, 'Video')
    require_file(args.model, 'Model')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    from cv.team_classifier import TeamClassifier
    from cv.tracker import PlayerTracker

    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    video_checksum = sha256_file(args.video)
    model_checksum = sha256_file(args.model)
    output_video = args.output_dir / 'tracked.mp4'

    print('=== TactiVision CV Observation Worker ===')
    print(f'Video: {args.video}')
    print(f'Model: {args.model}')

    tracker = PlayerTracker(model_path=str(args.model), conf=args.conf, device=args.device)
    tracking_data, ball_positions = tracker.track_video(str(args.video), str(output_video))

    write_json(args.output_dir / 'tracks.json', tracking_data)
    write_json(args.output_dir / 'ball_observations.json', ball_positions)

    classifier = TeamClassifier()
    cap = cv2.VideoCapture(str(args.video))
    calibration_frame = args.calibration_frame
    if calibration_frame is None:
        calibration_frame = max((det['frame'] for det in tracking_data), default=0) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame)
    ret, frame = cap.read()
    calibration_bboxes = [det['bbox'] for det in tracking_data if det['frame'] == calibration_frame]
    team_calibration_available = bool(ret and classifier.calibrate(frame, calibration_bboxes))

    sampled_teams: dict[int, list[str]] = {}
    for detection in tracking_data:
        if detection['frame'] % args.sample_frames != 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, detection['frame'])
        ret, frame = cap.read()
        if not ret:
            continue
        team = classifier.classify(frame, detection['bbox'])
        sampled_teams.setdefault(detection['track_id'], []).append(team)

    final_teams = {
        track_id: Counter(votes).most_common(1)[0][0]
        for track_id, votes in sampled_teams.items()
        if votes
    }
    for detection in tracking_data:
        detection['team'] = final_teams.get(detection['track_id'], 'unknown')
    cap.release()

    write_json(args.output_dir / 'tracks_with_teams.json', tracking_data)
    finished_at = datetime.now(timezone.utc)
    run_metadata = {
        'schemaVersion': 1,
        'runId': started_at.strftime('%Y%m%dT%H%M%SZ'),
        'status': 'observations-only',
        'startedAt': started_at.isoformat(),
        'finishedAt': finished_at.isoformat(),
        'durationSeconds': round(time.perf_counter() - started_clock, 3),
        'command': ' '.join(sys.argv),
        'python': platform.python_version(),
        'operatingSystem': platform.platform(),
        'video': {'path': str(args.video), 'checksum': video_checksum},
        'model': {'path': str(args.model), 'checksum': model_checksum},
        'parameters': {
            'confidence': args.conf,
            'device': args.device or 'auto',
            'sampleFrames': args.sample_frames,
            'calibrationFrame': calibration_frame,
        },
        'outputs': {
            'tracks': 'tracks.json',
            'tracksWithTeams': 'tracks_with_teams.json',
            'ballObservations': 'ball_observations.json',
            'annotatedVideo': 'tracked.mp4',
        },
        'quality': {
            'teamCalibration': 'available' if team_calibration_available else 'unavailable',
            'pitchCalibration': 'unavailable',
            'events': 'unavailable',
            'evidenceEligibility': 'candidate-only',
        },
    }
    write_json(args.output_dir / 'run.json', run_metadata)

    team_counts = Counter(final_teams.values())
    print('=== Team Assignment ===')
    print(f"  team_a: {team_counts.get('team_a', 0)} players")
    print(f"  team_b: {team_counts.get('team_b', 0)} players")
    print(f"  referee: {team_counts.get('referee', 0)} players")
    print(f"  unknown: {team_counts.get('unknown', 0)} players")
    print(f'Run metadata: {args.output_dir / "run.json"}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2) from error
