# TactiVision CV Module — Full Implementation Plan

> 12-Week Day-by-Day Execution Plan
> From local prototype to Google Cloud Run production deployment
> Based on: v2 Architecture PDF + GitHub mult-agents-football-match-analyzer (MIT)

---

## Prerequisites

- [x] Anaconda installed
- [x] GPU available (or Google Colab)
- [x] ffmpeg installed
- [x] Match video available (Chelsea vs Arsenal U18)
- [x] Day 1 done: YOLO tested on match video, ball detection issue noted
- [ ] Google Cloud account with Startups credits
- [ ] Roboflow account (free tier)

---

## PHASE 1: LOCAL PROTOTYPE (Weeks 1-3)

Goal: Full pipeline running locally — detection → tracking → team assignment → stats → JSON output

---

### Week 1: Detection + Tracking (CURRENT)

#### Day 1 ✅ DONE
- [x] Created conda env `tactiscout-cv`
- [x] Installed PyTorch + Ultralytics + OpenCV
- [x] Tested YOLOv8 and YOLOv11 on match video
- [x] Noted: ball detection poor, player detection good
- [x] Saved output videos to `runs/detect/`

---

#### Day 2 (Saturday) — Fork GitHub Repo + Annotation

**Morning (3 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Fork the mult-agents repo | `git clone https://github.com/predaaaasaaaaaaa/mult-agents-football-match-analyzer.git` into `C:\Users\Marouan\Documents\TactiScout-CV\reference\` | 5 min |
| 2 | Read their `src/agents/vision/` code | Understand how they call YOLO + ByteTrack | 30 min |
| 3 | Read their `src/utils/team_classifier.py` | Understand KMeans HSV approach | 20 min |
| 4 | Read their `src/utils/ball_interpolation.py` | **KEY: This solves your ball problem** | 20 min |
| 5 | Read their `src/agents/events/` | Understand pass/tackle/interception rules | 30 min |
| 6 | Create Roboflow account | https://roboflow.com — sign up free | 5 min |
| 7 | Extract 100 frames from YOUR match | `ffmpeg -i match.mp4 -vf "fps=1/30" -q:v 2 annotation_frames/frame_%04d.jpg` | 5 min |
| 8 | Upload 50 frames to Roboflow | New project: "TactiVision-Football" | 10 min |

**Afternoon (3 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 9 | Annotate 50 frames in Roboflow | Classes: `player`, `ball`, `goalkeeper`, `referee` | 2.5 hrs |
| 10 | Focus on ball annotations | Mark ball even when blurry/small/partially hidden | Included |
| 11 | Export dataset in YOLOv8 format | Download ZIP | 5 min |
| 12 | Unzip to `datasets/football-v1/` | Check train/val/test folders exist | 5 min |

**Annotation rules:**
- Draw tight boxes head-to-feet for players
- Ball: mark even when 10px wide or motion-blurred
- Goalkeeper: separate class (different jersey color)
- Referee: separate class
- Skip replay/graphic frames

**End of Day 2 Deliverable:**
- Reference repo cloned and studied
- 50 annotated frames ready for training
- Understanding of how team classification + ball interpolation works

---

#### Day 3 (Sunday) — Fine-Tune + Ball Interpolation

**Morning (2 hours)**

| # | Task | Command | Time |
|---|------|---------|------|
| 1 | Annotate 50 more frames in Roboflow | Total: 100 annotated | 1.5 hrs |
| 2 | Export full dataset (YOLO format) | Download updated ZIP | 5 min |
| 3 | Create `data.yaml` for training | See below | 5 min |

**`C:\Users\Marouan\Documents\TactiScout-CV\data.yaml`:**
```yaml
path: C:/Users/Marouan/Documents/TactiScout-CV/datasets/football-v1
train: train/images
val: val/images
test: test/images

nc: 4
names: ['player', 'ball', 'goalkeeper', 'referee']
```

| # | Task | Command | Time |
|---|------|---------|------|
| 4 | Start fine-tuning YOLOv8m | `yolo train model=yolov8m.pt data=data.yaml epochs=50 imgsz=1280 batch=4` | Let it run (~1-2 hrs) |

**Afternoon (3 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 5 | While training: create project structure | See below | 30 min |
| 6 | Copy ball_interpolation.py from reference repo | Adapt to our structure | 30 min |
| 7 | Write `cv/detector.py` | From the v2 PDF Step 3 code | 30 min |
| 8 | Check training results | Open `runs/detect/train/results.png` | 10 min |
| 9 | Test fine-tuned model on match | `yolo predict model=runs/detect/train/weights/best.pt source="match.mp4" save=True conf=0.2` | 20 min |
| 10 | Compare ball detection: base vs fine-tuned | Count ball detections in both outputs | 30 min |

**Project structure to create:**
```
C:\Users\Marouan\Documents\TactiScout-CV\
├── cv/
│   ├── __init__.py
│   ├── detector.py          # From v2 PDF
│   ├── tracker.py           # Day 4
│   ├── team_classifier.py   # Day 5 (from reference repo)
│   ├── ball_interpolation.py # Day 3 (from reference repo)
│   ├── homography.py        # Day 6
│   ├── event_detector.py    # Day 8
│   ├── stats_engine.py      # Day 9
│   ├── formatter.py         # Day 10
│   └── pipeline.py          # Day 11
├── datasets/
│   └── football-v1/
├── models/
│   ├── yolov8m.pt
│   └── best.pt              # After fine-tuning
├── data/
│   ├── match.mp4
│   └── cache/
├── output/
├── reference/               # Cloned GitHub repo
├── data.yaml
├── requirements.txt
└── run_pipeline.py
```

**End of Day 3 Deliverable:**
- 100 annotated frames, model training started/finished
- Ball detection improvement measured (expect 20% → 60%+)
- Project structure created
- `detector.py` and `ball_interpolation.py` written

---

#### Day 4 (Monday) — ByteTrack Tracking

**Morning (3 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Install supervision + lap | `pip install supervision lap` | 5 min |
| 2 | Write `cv/tracker.py` | Use the v2 PDF code (Step 4) | 1 hr |
| 3 | Write `test_tracking.py` | Process 5-min clip, output tracked video | 1 hr |
| 4 | Run tracking on match clip | `python test_tracking.py` | 30 min |
| 5 | Watch output video | Are IDs consistent? Count unique players | 30 min |

**`test_tracking.py`:**
```python
import cv2
import supervision as sv
from cv.detector import PlayerDetector
from cv.tracker import PlayerTracker
import json

detector = PlayerDetector("models/best.pt")  # fine-tuned
tracker = PlayerTracker()

video_path = "data/match_5min.mp4"  # ffmpeg -i match.mp4 -t 300 -c copy match_5min.mp4
video_info = sv.VideoInfo.from_video_path(video_path)
tracking_data = []

with sv.VideoSink("output/tracked.mp4", video_info) as sink:
    for idx, frame in enumerate(sv.get_video_frames_generator(video_path)):
        detections = detector.detect(frame)
        players = tracker.update(detections, frame)
        
        for p in players:
            tracking_data.append({
                "frame": idx,
                "player_id": p.track_id,
                "team": p.team,
                "x": p.pixel_pos[0],
                "y": p.pixel_pos[1],
                "speed": p.speed,
                "bbox": p.bbox
            })
        
        # Annotate frame
        labels = [f"{p.team}#{p.track_id}" for p in players]
        # ... draw boxes and labels
        sink.write_frame(frame)
        
        if idx % 100 == 0:
            print(f"Frame {idx}/{video_info.total_frames}")

with open("output/tracking.json", "w") as f:
    json.dump(tracking_data, f)

print(f"Done. {len(tracking_data)} detections. Unique IDs: {len(set(d['player_id'] for d in tracking_data))}")
```

**Afternoon (2 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 6 | Analyse tracking output | How many unique IDs? Should be ~22-25 | 30 min |
| 7 | Tune tracker parameters | `lost_track_buffer`, `minimum_matching_threshold` | 45 min |
| 8 | Re-run with tuned params | Compare ID count | 30 min |
| 9 | Document issues | ID switches? Lost tracks? False detections? | 15 min |

**End of Day 4 Deliverable:**
- `tracker.py` working
- Tracked video output with player IDs
- `tracking.json` with all positions per frame
- Unique player count (target: 18-25)
- List of tracking issues to fix later

---

#### Day 5 (Tuesday) — Team Classification + Physical Stats

**Morning (2.5 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Write `cv/team_classifier.py` | Adapt from reference repo (KMeans on HSV torso crops) | 1 hr |
| 2 | Test team assignment on 5-min clip | Run classifier on tracked players | 30 min |
| 3 | Verify visually | Team A = one color, Team B = another? | 30 min |
| 4 | Handle edge cases | Goalkeeper (different jersey), referee (yellow/black) | 30 min |

**Team classification approach (from reference repo + v2 PDF):**
```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamClassifier:
    def __init__(self):
        self.team_colors = None  # Calibrated after first 100 frames
        self.calibrated = False
    
    def calibrate(self, frame_crops: list):
        """Run once on first 100 player crops to learn team colors."""
        colors = []
        for crop in frame_crops:
            h, w = crop.shape[:2]
            torso = crop[int(h*0.15):int(h*0.45), :]
            hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
            median_color = np.median(hsv.reshape(-1, 3), axis=0)
            colors.append(median_color)
        
        # Cluster into 3 groups: team_a, team_b, referee
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        km.fit(colors)
        self.team_colors = km.cluster_centers_
        self.calibrated = True
    
    def classify(self, crop) -> str:
        """Classify a single player crop into home/away/referee."""
        if not self.calibrated:
            return "unknown"
        h, w = crop.shape[:2]
        torso = crop[int(h*0.15):int(h*0.45), :]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        median_color = np.median(hsv.reshape(-1, 3), axis=0)
        
        # Find closest cluster
        distances = [np.linalg.norm(median_color - c) for c in self.team_colors]
        label = np.argmin(distances)
        
        # Map labels (sorted by hue) to team names
        return ["home", "away", "referee"][label]
```

**Afternoon (2.5 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 5 | Write `cv/stats_engine.py` (basic) | Distance + speed per player from tracking data | 1 hr |
| 6 | Compute possession | Which team's player closest to ball per frame | 30 min |
| 7 | Run full stats on 5-min clip | Print per-player stats table | 30 min |
| 8 | Validate: speeds reasonable? | Should be 0-35 km/h (in pixels initially) | 30 min |

**Stats to compute (from reference repo):**
```python
# Per player:
# - Total distance (sum of position deltas)
# - Top speed (max speed across all frames)
# - Sprint count (consecutive frames above sprint threshold)
# - Average position (for heatmap)
# - Time on ball (frames where closest to ball)

# Per team:
# - Possession % (frames where team has ball)
# - Pass count (from event detection — Day 8)
# - Formation shape (from average positions)
```

**End of Day 5 Deliverable:**
- Teams correctly assigned by jersey color
- Per-player stats: distance, top speed, sprint count
- Per-team possession %
- Validated: numbers are reasonable (not negative, not >35 km/h)

---

#### Day 6 (Wednesday) — Pitch Homography

**Morning (3 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Write `cv/homography.py` | From v2 PDF (RANSAC version) | 1 hr |
| 2 | Extract one clear frame with visible pitch lines | `ffmpeg -i match.mp4 -ss 00:05:00 -frames:v 1 pitch_frame.jpg` | 5 min |
| 3 | Manually identify 4+ pitch landmarks | Use an image viewer, note pixel coordinates | 30 min |
| 4 | Define real-world coordinates for those landmarks | Penalty box corners = known meters | 15 min |
| 5 | Compute homography matrix | `cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)` | 15 min |
| 6 | Test: transform a player position | Should be within 0-105m x 0-68m | 15 min |
| 7 | Transform ALL tracking data to meters | Re-compute from pixel positions | 30 min |

**Key pitch landmarks (meters on a 105m x 68m pitch):**
```python
# Penalty box left: top-left corner = (0, 13.84)
# Penalty box left: top-right corner = (16.5, 13.84)
# Penalty box left: bottom-right corner = (16.5, 54.16)
# Centre circle top = (52.5, 24.84)
# Centre spot = (52.5, 34.0)
# Penalty spot left = (11, 34.0)
# Penalty spot right = (94, 34.0)
```

**Afternoon (2 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 8 | Recompute speed in m/s and km/h | Using real meter positions now | 30 min |
| 9 | Recompute distance in km | Sum of meter deltas | 15 min |
| 10 | Validate: distance should be 0.5-1.5 km per 5 min | ~10 km per 90 min | 15 min |
| 11 | Generate player position heatmap | matplotlib scatter on pitch outline | 45 min |
| 12 | Detect average formation shape | Mean position per player → 4-3-3? | 15 min |

**End of Day 6 Deliverable:**
- Homography working: pixel → real meters
- Speed in km/h (validate: 0-35 range)
- Distance in km (validate: 0.5-1.5 per 5 min)
- Position heatmap generated
- Formation shape detected

---

#### Day 7 (Thursday) — Full Pipeline v0.1 + Review

**Morning (2 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Write `cv/pipeline.py` | Orchestrate: detect → track → classify → homography → stats | 1.5 hrs |
| 2 | Write `run_pipeline.py` | Entry point: `python run_pipeline.py data/match_5min.mp4` | 30 min |

**Afternoon (3 hours)**

| # | Task | Details | Time |
|---|------|---------|------|
| 3 | Run full pipeline on 5-min clip | End-to-end | 30 min |
| 4 | Review all output | Stats make sense? Teams correct? Speed valid? | 30 min |
| 5 | Fix top 3 bugs found | Whatever broke | 1 hr |
| 6 | Write "Week 1 Results" document | What works, metrics, issues | 30 min |
| 7 | Plan Week 2 based on findings | What needs most attention? | 30 min |

**End of Week 1 Deliverable:**
- Full local pipeline: video → detection → tracking → team → homography → stats
- JSON output with per-player statistics
- Identified issues for Week 2
- Comparison: our results vs reference repo results

---

### Week 2: Event Detection + StatsBomb Output

#### Day 8 (Friday) — Event Detection

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Write `cv/event_detector.py` | From v2 PDF (pressure, sprint, formation) | 2 hrs |
| 2 | Add pass detection rule | Ball changes possession (nearest player on same team → different player same team) | 1.5 hrs |
| 3 | Add tackle detection rule | Ball changes from team A to team B player | 1 hr |
| 4 | Test on 5-min clip | Count events detected | 30 min |
| 5 | Validate: reasonable event counts? | ~20-40 passes per 5 min is normal | 30 min |

---

#### Day 9 (Saturday) — Advanced Stats + Caching

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Add per-video caching | Save YOLO results to JSON, skip re-detection on rerun | 1.5 hrs |
| 2 | Compute advanced stats | Passes completed, interceptions, tackles per player | 1.5 hrs |
| 3 | Add ball interpolation to pipeline | Fill gaps between detected ball positions | 1 hr |
| 4 | Measure ball coverage improvement | Before interpolation vs after (target: 20% → 50%+) | 30 min |

---

#### Day 10 (Sunday) — StatsBomb JSON Output

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Write `cv/formatter.py` | From v2 PDF Step 5 — StatsBomb JSON schema | 1.5 hrs |
| 2 | Generate StatsBomb JSON for your match | Run pipeline → format output | 30 min |
| 3 | Validate JSON structure | Compare to real StatsBomb Open Data format | 1 hr |
| 4 | Write `cv/visualizer.py` | Generate heatmaps + annotated video | 2 hrs |

---

#### Day 11 (Monday) — Full Match Processing

| # | Task | Details | Time |
|---|------|---------|------|
| 1 | Process full 45-min half | Run pipeline on first half | Let it run |
| 2 | Implement video chunking (local) | Split into 5-min chunks, process each, stitch results | 2 hrs |
| 3 | Compare: single vs chunked | Same output? IDs consistent across chunks? | 1 hr |
| 4 | Fix ID bridging across chunks | Player ID continuity at chunk boundaries | 1.5 hrs |

---

#### Day 12-14 (Tue-Thu) — Testing + Bug Fixing

| Day | Focus | Tasks |
|-----|-------|-------|
| 12 | Accuracy testing | Run on 3 different match clips. Measure detection rate, tracking stability, event accuracy |
| 13 | Bug fixing | Fix top issues from Day 12 testing |
| 14 | Documentation | Write README, document API, create requirements.txt |

**End of Week 2 Deliverable:**
- Full pipeline with event detection + StatsBomb JSON output
- Video chunking working (parallel-ready)
- Ball interpolation improving coverage to 50%+
- Tested on 3+ match clips
- Per-video caching (instant re-analysis)

---

### Week 3: Dockerization + Local Polish

#### Day 15-16 — Docker

| # | Task | Details |
|---|------|---------|
| 1 | Create `Dockerfile` | From v2 PDF Step 9 — NVIDIA CUDA base image |
| 2 | Create `requirements.txt` | Pin all versions |
| 3 | Create `docker-compose.yml` | Mount volumes for data/ and output/ |
| 4 | Build and test Docker image | `docker build -t tactivision-cv .` |
| 5 | Run pipeline in Docker | Verify same output as local |

#### Day 17-18 — FastAPI

| # | Task | Details |
|---|------|---------|
| 1 | Write `api/cv_routes.py` | From v2 PDF Step 7 — upload + process + status + results |
| 2 | Add `/health` endpoint | For Cloud Run health checks |
| 3 | Test API locally | Upload video via curl, poll status, get JSON |
| 4 | Add Firestore job tracking (mock) | Status: processing/done/failed |

#### Day 19-21 — Integration Testing

| # | Task | Details |
|---|------|---------|
| 1 | Test Docker + API end-to-end | Upload → process → get results via HTTP |
| 2 | Test on 3 different match formats | Different resolutions, lighting, camera angles |
| 3 | Performance benchmark | Time per frame, total time for 5-min clip |
| 4 | Memory profiling | Does it stay within 16GB Cloud Run limit? |

**End of Week 3 Deliverable:**
- Dockerized pipeline
- FastAPI serving video analysis
- Tested on multiple match clips
- Performance numbers documented
- Ready for cloud deployment

---

## PHASE 2: GOOGLE CLOUD DEPLOYMENT (Weeks 4-6)

#### Day 22-23 — GCP Setup

| # | Task | Command |
|---|------|---------|
| 1 | Install gcloud CLI | `winget install Google.CloudSDK` |
| 2 | Login + set project | `gcloud auth login && gcloud config set project tactivision-ai` |
| 3 | Enable required APIs | Compute Engine, Cloud Run, Cloud Storage, Cloud Tasks, Firestore, Artifact Registry |
| 4 | Create Storage buckets | `gsutil mb gs://tactivision-videos && gsutil mb gs://tactivision-outputs` |
| 5 | Create Firestore database | Via GCP Console |
| 6 | Create Artifact Registry | For Docker images |

#### Day 24-25 — Deploy Single-Chunk to Cloud Run

| # | Task | Command |
|---|------|---------|
| 1 | Push Docker image to Artifact Registry | `gcloud builds submit --tag gcr.io/tactivision-ai/cv-module:v1 .` |
| 2 | Deploy to Cloud Run (no GPU first) | Test on CPU to verify API works |
| 3 | Test: upload small clip via API | `curl -X POST .../cv/analyze -F video=@clip.mp4` |
| 4 | Add GPU | `gcloud run deploy ... --gpu 1 --gpu-type nvidia-l4` |
| 5 | Re-test with GPU | Verify speed improvement |

#### Day 26-28 — Add Parallel Chunking

| # | Task | Details |
|---|------|---------|
| 1 | Write `cv/pre_processor.py` | From v2 PDF — ffmpeg chunking + Cloud Tasks dispatch |
| 2 | Create Cloud Tasks queue | `gcloud tasks queues create cv-processing --location=europe-west4` |
| 3 | Add `/process-chunk` endpoint | Internal endpoint called by Cloud Tasks |
| 4 | Write `cv/stitcher.py` | Merge chunk JSONs + bridge player IDs |
| 5 | Test full 90-min match | Upload → chunk → parallel process → stitch → done |
| 6 | Measure: total time for 90 min | Target: < 5 minutes |

#### Day 29-31 — Production Hardening

| # | Task | Details |
|---|------|---------|
| 1 | Add error handling | Retry failed chunks, dead letter queue |
| 2 | Add Cloud Monitoring | Track latency, GPU util, error rates |
| 3 | Add authentication | Service accounts, IAM roles |
| 4 | Cost monitoring | Budget alerts at $50/month |
| 5 | Load test | Process 5 matches simultaneously |

**End of Phase 2 Deliverable:**
- Cloud Run deployment with GPU
- Parallel video chunking (90 min → 5 min processing)
- Full match StatsBomb JSON output
- Monitoring + error handling
- Cost: ~$0.50 per match

---

## PHASE 3: QUALITY + RE-ID (Weeks 7-9)

#### Week 7: Visual Re-ID

| Day | Task |
|-----|------|
| 36 | Install torchreid, study OSNet architecture |
| 37 | Write `cv/reid_cache.py` from v2 PDF |
| 38 | Integrate Re-ID into tracker.py |
| 39 | Test: measure ID swap reduction |
| 40 | Tune cosine similarity threshold (0.65) |
| 41-42 | Test on 5 full matches, document improvement |

#### Week 8: RANSAC Homography

| Day | Task |
|-----|------|
| 43 | Implement RANSAC filtering in homography.py |
| 44 | Add H_prev fallback (freeze when <4 landmarks) |
| 45 | Test on panning/zooming footage |
| 46 | Add auto-recalibration every N frames |
| 47-49 | Test stability across full matches |

#### Week 9: Fine-Tuning + More Events

| Day | Task |
|-----|------|
| 50 | Annotate 200 more frames (total: 300) |
| 51 | Retrain YOLO with larger dataset |
| 52 | Add shot detection (ball velocity towards goal) |
| 53 | Add dribble detection (player moves with ball past opponent) |
| 54 | Add cross detection (ball from wide area into box) |
| 55-56 | Validate all events against manual annotation |

---

## PHASE 4: RAG INTEGRATION + PRODUCTION (Weeks 10-12)

#### Week 10: TactiVision RAG Integration

| Day | Task |
|-----|------|
| 57 | Write `integration/cv_to_rag.py` from v2 PDF |
| 58 | Test: CV output → RAG ingestion → Opposition Agent query |
| 59 | Verify: Agent answers reference real CV-detected events |
| 60-63 | End-to-end test: upload video → agent produces tactical report |

#### Week 11: Dashboard + UX

| Day | Task |
|-----|------|
| 64 | Add video upload to TactiVision web dashboard |
| 65 | Add progress bar (chunk completion tracking) |
| 66 | Add result visualization (heatmaps, stats tables) |
| 67-70 | Polish + user testing |

#### Week 12: Launch Prep

| Day | Task |
|-----|------|
| 71 | Security audit (auth, data isolation per customer) |
| 72 | Performance final benchmark (100 matches/month capacity) |
| 73 | Cost optimization (min-instances=0, preemptible VMs) |
| 74 | Documentation finalization |
| 75 | License check (RT-DETR swap if needed for production) |
| 76-77 | Beta launch: first 5 clubs |

---

## Success Metrics

| Metric | Week 3 | Week 6 | Week 12 |
|--------|--------|--------|---------|
| Player Detection mAP50 | 0.70 | 0.75 | 0.85 |
| Ball Detection | 50%+ (with interpolation) | 55% | 65% |
| Tracking MOTA | 0.55 | 0.65 | 0.75 |
| Homography Error | <3m | <2m | <1.5m |
| ID Swap Rate | <10/match | <5/match | <2/match |
| Processing Time (90 min) | 30 min (local) | 5 min (cloud) | 3 min (optimized) |
| Events vs manual annotation | 50% match | 65% match | 80% match |
| Cost per match | $0 (local) | $0.50 | $0.40 |

---

## Immediate Next Step (Day 2 — Tomorrow)

1. Fork the GitHub repo
2. Start annotating frames in Roboflow
3. Study their ball interpolation code (it's the key to solving your Day 1 ball detection problem)
