# Stage 2d: the world-space tracker

Stage 2d replaced ByteTrack's IDs and the per-ID Kalman filter with our own tracker in world coordinates. This page records how the new tracker scores, which gates it passes, and the one it does not.

**Result:**
- Every fixture gate passes, and CI checks them.
- The live bag replay (step 3) **did not pass**. One run missed time to DYNAMIC, and another missed ID switches. Both misses come from too few detections, not from the tracker, so their gates move to stage 2c. The investigation is below.

## What changed

[tracking.py](../r_one/r_one/tracking.py) holds the tracker:
- **Filter.** Each track has a constant-velocity Kalman filter over `[x, y, vx, vy]`. It uses the real time between frames, and its process noise grows with that time.
- **Matching.** Tracks are matched to measurements by Mahalanobis distance (99% gate), with a 1.5 m pre-check. Classes must agree, and `scipy.optimize.linear_sum_assignment` solves the matching.
- **Lifecycle.** A track is published after 2 matches. It coasts for up to 2.5 s without a match, then ends. At most 64 tracks are kept.
- **Motion.** A second, slower filter per track decides STATIC or DYNAMIC. The main filter's velocity is too noisy to ever prove that an object stands still.
  - DYNAMIC when `speed − 1.0·σv > speed_threshold`.
  - STATIC after 5 slow matches in a row.

[world_model.py](../r_one/r_one/world_model.py) stores confirmed STATIC tracks that are at least 1 s old, and holds at most 256 entries. A track that turns DYNAMIC is removed.

The `tracking_filter` and `ema_alpha` parameters were removed. `speed_threshold` keeps its meaning.

## How it was measured

- **Fixture.** The saved detections from [baseline_2a.md](baseline_2a.md) are replayed through the tracker with [replay_tracker.py](../tools/replay_tracker.py). `TestFixtureGates` in [test_tracking.py](../r_one/test/test_tracking.py) does this on every test run.
- **Live replay.** The sample bag is played through the full node in the Docker image, with the command in [baseline_2a.md](baseline_2a.md) plus `-e ROS_LOCALHOST_ONLY=1`. It was run three times:
  - run 1;
  - run A;
  - run B, with `OPENBLAS_NUM_THREADS=1`.
- **Scoring.** Every run was scored with [score_run.py](../tools/score_run.py).

The live runs are the check on data the tracker was not tuned on. Their logs hold the tracker's filtered positions, so they cannot be replayed offline.

## Results

| Metric | Fixture, before (2a) | Fixture, 2d | Run 1 | Run A | Run B | Gate |
|---|---|---|---|---|---|---|
| Frames | 69 | 69 | 20 | 56 | 71 | |
| Walker ID switches | 14 | **0** | 2 ✗ | 0 | 0 | 0 fixture, ≤ 1 live |
| Walker recall | 0.884 | 0.855 | 0.500 | 0.732 | 0.817 | ≥ 0.85 fixture |
| Position RMSE (m) | 0.214 | 0.215 | 0.206 | 0.216 | 0.216 | ≤ 0.25 |
| Velocity RMSE, cruising (m/s) | 0.499 | **0.148** | 0.375 ✗ | 0.212 | 0.161 | ≤ 0.25 |
| Velocity RMSE, overall (m/s) | 0.531 | 0.294 | 0.436 | 0.338 | 0.282 | ≤ 0.35 |
| False DYNAMIC rate | 0 | 0 | 0 | 0 | 0 | ≤ 1% |
| Time to DYNAMIC (s) | 1.26 | 1.16 | 1.10 | 2.46 ✗ | 0.66 | ≤ 1.5 |
| Ghosts | 1 | **0** | 0 | 0 | 0 | 0 |
| Rate, p95 (Hz) | 1.66 | 1.66 | 0.22 | 1.00 | 2.00 | ≥ 2 (stage 2c) |
| Latency, p95 (s) | 0.182 | 0.182 | 0.327 | 0.173 | 0.302 | |

Notes on the table:
- **Ghosts.** Every live run ends with the world model holding only the standing person, at about (4.33, −1.76).
- **Tracker cost.** On the fixture, one tracker step takes 0.37 ms median and 0.58 ms p95. This was measured on the host with a pure-Python stand-in for scipy's matcher, so the container should be no slower.
- **Walker recall** stays the stage 2c gate (≥ 0.90). It depends on detections and frame rate, not on the tracker.
- **Overall velocity** can't reach 0.25 for any causal tracker on this script: the walker reverses instantly. It is a no-regression bound at 0.35, and cruising velocity carries the 0.25 gate.

### Time to DYNAMIC was redefined in this stage

`time_to_dynamic` now counts from the later of two times: the track's first frame, or the walker's last movement start. The old definition counted from the first frame, so a track first seen during a pause or turn was charged for time the walker spent standing.

Under the new definition the 2a baseline fixture scores 1.26 s instead of 11.9 s. [baseline_2a.md](baseline_2a.md) keeps the old figure as its historical record.

## Why the live replay did not pass

### Run A: time to DYNAMIC 2.46 s

- **What happened.** The walker was first seen at 39.57 s, just before its 0.5 s pause at the end of its path. It turned back at 40.5 s. The node then got 4 detections in 0.66 s, none for 1.8 s, and the track turned DYNAMIC on the next detection, at 42.97 s.
- **The other runs never tested this.** In runs 1 and B the walker was first seen mid-walk. Run B's pass therefore does not cover run A's case.
- **It repeats offline.** A synthetic walker detected at run A's exact times shows the same delay in 9 of 10 seeds.
- **Real noise is lower than the tracker assumes.** The fixture's raw positions vary by (0.046, 0.107) m for the walker and (0.004, 0.047) m for the standing person. The tracker is configured for 0.2 m.
- **Four kinds of change to the motion decision were tested, with several settings each, and none passes.** Each was tried on run A's case, on a walker first seen anywhere on its path at 5 Hz and 2 Hz, on a standing person, and on the fixture.

  | Change | Run A's case fails | Standing false DYNAMIC (5 Hz, 0.13 m) |
  |---|---|---|
  | None (current) | 73–93% | 0.18% |
  | Faster slow filter | 90% | 0.18%, but STATIC takes up to 16 s |
  | Speed fitted over the last 0.6–1.0 s | 75–90% | 0.18–1.1% (3–23% at 0.2 m noise) |
  | Speed fitted over the last 4–5 points | 20–47% | **4–16%** |
  | More velocity uncertainty after a large innovation | 73–93% | 0.18% (3–29% at 0.2 m noise) |

- **No rule can reliably pass this case.** Take an ideal rule that knows exactly when the walker turned, uses only the 4 detections after the turn, and is allowed 1% false alarms. It catches the walker only **68% of the time at 0.1 m noise, and 39% at 0.13 m**. A real tracker can do no better.

The miss comes from too few detections in the 1.5 s after the turn. Stage 2c addresses that: it targets frame rate and dropped frames.

### Run 1: two ID switches

The node processed only 20 frames, at 0.22 Hz p95. The walker went undetected for 6.6 s (41.87 → 48.47 s), and then for 3.6 s. Both gaps are longer than the 2.5 s coasting limit, so ending the track and starting a new ID is correct behaviour.

The low frame rate in run 1 was not explained. Run B, with one OpenBLAS thread, processed more frames but was slower per frame than run A, so thread contention is not supported as the cause.

### What was decided

- **Targets.** The 1.5 s and ≤ 1 switch targets stay as they are.
- **Gate.** They become part of stage 2c's gate: on 3 live runs, ID switches ≤ 1 and time to DYNAMIC ≤ 1.5 s on every run.
- **Regression tests.** Two tests in `TestMotion` lock in what the tracker guarantees today:
  - **`test_walker_first_seen_at_a_turn`** uses run A's detection times. The track keeps one ID, never turns STATIC, and turns DYNAMIC no later than the first detection after the gap.
  - **`test_walker_first_seen_anywhere_on_its_path`** starts the track every 0.1 s along the path:
    - at 5 Hz, every start turns DYNAMIC within 1.5 s (worst 1.3 s);
    - at 2 Hz the bound is 1.8 s (worst 1.7 s).

    The 2 Hz bound guards against regressions. It fails when the slow filter is made slower, when the DYNAMIC margin is raised, or when coasting is shortened.

## Known limits

- **At 2 Hz, a walker first seen near a turn can take up to 2.2 s to turn DYNAMIC.** In wider sweeps about 5% of start points are slower than 1.5 s. At 5 Hz none are.
- **The configured measurement noise (0.2 m) is about twice what the fixture shows.** A tracker tuned to the real noise could decide faster. That is a lead for stage 2c, not a change made here, because the tuned values stay fixed until a stage measures a better set.
- **UNKNOWN tracks are drawn green in RViz,** the same as STATIC tracks.
- **The container's scipy is the apt package, built for NumPy < 1.25.** The image runs NumPy 1.26.4. scipy prints a version warning but works, and every test passes in the container. A future image change could break this import.

## Changes from the stage plan

- **Slow motion filter.** `motion_accel_std_mps2` is 0.1 and `motion_initial_velocity_std_mps` is 0.3. `motion_sigma_k` is 1.0 instead of the planned 0.5, and it applies to DYNAMIC only. With one filter, the velocity uncertainty settled near 0.53 m/s, so STATIC was unreachable.
- **[messages.py](../r_one/r_one/messages.py).** Added `append_delete_markers`, plus the `TRACK_MARKER_NAMESPACES` and `WORLD_MODEL_NAMESPACE` constants. Ended tracks and shrunk world-model entries are deleted from RViz with these.
- **Costmap.** Coasting DYNAMIC tracks now get the same 0.25 m ring of points around each predicted position as matched tracks. Before, they got only the centre points.
- **[run_log.py](../r_one/r_one/run_log.py).** Added `track_record(track)`.
- **README.** Beyond the planned edits, it also updates the architecture diagram, the threading note, the `active_tracks` description and the status table.
- **Tests.**
  - `TestFrameTracking` in [test_spatial_vision_guards.py](../r_one/test/test_spatial_vision_guards.py) runs two frames through the node. It checks that a detection is published only after its track is confirmed.
  - The two regression tests above are added in [test_tracking.py](../r_one/test/test_tracking.py).
