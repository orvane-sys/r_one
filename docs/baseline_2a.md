# Baseline: perception before the tracker rewrite

This records how the current code (ByteTrack plus a per-ID Kalman filter, from `v0.1.0`) scores on the sample bag. Later stages are measured against it.

## How it was measured

- **Input:** `sample_bags/sample_spatial_run` from the `v0.1.0` release. It is 23.1 s of Gazebo data at 10 Hz.
- **Settings:** the bag launch defaults: `depth_source:=depth_image`, `process_every_n_frames: 2` and `img_size: 320`.
- **Machine:** a laptop CPU inside the project's Docker image, with no GPU.
- **Run log:** saved as [sample_run_baseline.jsonl](../r_one/test/fixtures/sample_run_baseline.jsonl). It has 69 frames.

To reproduce:

```bash
mkdir -p reports
docker compose run --rm --no-deps -v "$PWD/sample_bags:/data:ro" -v "$PWD/reports:/out" r_one \
  ros2 launch r_one_bringup spatial_bag_playback.launch.py \
    bag_path:=/data/sample_spatial_run rviz:=false loop:=false run_log_path:=/out/run.jsonl
# Press Ctrl+C after ros2 bag exits
python3 tools/score_run.py reports/run.jsonl --gates
```

## Ground truth check

The actors' positions come from the waypoints in `spatial_demo.world`, the message stamps and the robot's spawn offset. We needed to know that this truth matches what the camera saw before trusting any score:

- The best time shift between the stamps and the walker's script is **0.00 s**. The fit cost is 3.2 there, compared with 10.7 at ±0.5 s and 32 at ±1 s, so the match is clear.
- **Every** person detection lies within 0.5 m of an actor.
- The standing person has a mean error of 0.19 m, close to the walker's 0.23 m, so the spawn offset is right.

`TestSampleRun` in [test_scoring.py](../r_one/test/test_scoring.py) keeps these checks running in CI.

## Scores

| Metric | Baseline | Target | Result |
|---|---|---|---|
| Walker ID switches | 14 in 22.7 s (61.7 per 100 s) | ≤ 1 per 100 s | Fail |
| Walker recall | 88.4% | ≥ 90% | Fail |
| Position RMSE | 0.214 m | ≤ 0.25 m | Pass |
| Velocity RMSE | 0.531 m/s (0.499 while cruising) | ≤ 0.25 m/s | Fail |
| False DYNAMIC on the standing person | 0% | ≤ 1% | Pass |
| Time to DYNAMIC | 11.9 s (slowest ID) | ≤ 1.5 s | Fail |
| Ghosts after the run | 1 | 0 | Fail |
| Rate at p95 | 1.66 Hz | ≥ 2 Hz | Fail |
| Latency at p95 | 0.18 s | none | — |

## What causes the failures

**ID switches.** The walker gets a new ID almost every time it is missed for a single frame. That gives 8 different IDs in 23 s. ByteTrack also reuses old IDs: ID 2 belongs to the walker at the start and returns three times later. Stage 2d's world-space tracker is meant to fix this.

**Velocity.** Each new ID starts its Kalman filter at zero speed, and the speed takes 3 to 5 frames to climb towards 0.75 m/s. The error while cruising (0.50 m/s) is almost the same as the error overall (0.53 m/s), so the cause is ID churn, not the walker's turns.

**Time to DYNAMIC.** A typical new ID becomes DYNAMIC on its second or third frame, which is 0.2 to 0.5 s. The 11.9 s figure belongs to ID 2. It was first seen at the start, lost, and only reported DYNAMIC after it was reused 12 s later. This metric takes the slowest ID, so it shows ID reuse as well as slow motion detection.

**Ghost.** The walker's ID 2 measured 0.10 to 0.14 m/s for two frames, which is below the 0.15 m/s threshold. That frame was stored in the world model as static at (4.28, -0.06), on the walker's path. It was never removed because ID 2 was not seen again. Also, every new ID is STATIC on its first frame, because its speed starts at zero, so for a moment each one joins the world model. This is bug H7, which stage 2d fixes.

**Rate.** Most frames arrive 0.2 s apart (5 Hz), as configured. The p95 has two causes:
- The first inference took 5.2 s to warm up, which left a 5 s gap in the log. The 8 s `start_delay` covers loading the model but not this first call.
- The slowest frames took 0.18 to 0.26 s, longer than the 0.2 s between frames, so some frames were dropped.

Stage 2c's single worker thread with stale-frame dropping addresses the second cause. A warm-up inference at startup would remove the first.

**Recall.** The walker is missing from 8 of the 69 frames. The log cannot tell whether the detector missed it or the depth lookup failed, because frames without a range are not logged. Frames the node skipped entirely also do not count against recall. Stage 2c should log both kinds of drop.

## Limits of this baseline

- **ID switches:** the bag is only 22.7 s long, so one switch counts as 4.4 per 100 s. A fair ID-switch score needs a recording of at least 100 s.
- **Detector:** this is a single run on one machine with one detector (`yolov8n.pt`). Stage 4a replaces the detector, and the targets will be checked again then.
