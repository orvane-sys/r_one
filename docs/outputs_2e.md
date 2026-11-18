# Stage 2e: outputs

Stage 2e publishes the tracker's state on `perception/tracks`, keeps every output current on every frame, sizes each object from its detection box, and makes the diagnostics report the real processing rate. This page records what changed, and why the costmap-clearing gate moved to stage 4c.

## What changed

**`perception/tracks` (E2, E3).**
- A new `r_one_msgs/TrackArray` topic, relative like the other outputs, so it becomes `/robot_1/perception/tracks` under a namespace. QoS: Reliable, Volatile, depth 10.
- It carries every live track, with its lifecycle `state`:
  - `TENTATIVE` before the track's second match;
  - `CONFIRMED` when this frame matched it;
  - `COASTING` when it was not matched and its position is predicted.
- Each track has its `motion`, position, velocity, size, the 4 × 4 covariance of [x, y, vx, vy] (row-major), `age_s` and `time_since_update_s`. Consumers filter by state.
- Markers, `detections_3d` and the run log keep the previous set: confirmed tracks matched in this frame.
- `Tracker.lifecycle(track)` in [tracking.py](../r_one/r_one/tracking.py) decides the state. [messages.py](../r_one/r_one/messages.py) `build_track_array` fills the message.

**Outputs on every frame (H5, E1).**
- Before, the costmap cloud and `detections_3d` were published only when they had something in them. After the walker left, a consumer kept the last non-empty message.
- Now the cloud, `detections_3d` and `perception/tracks` are published on every processed frame, empty when there is nothing.

**Measured box size (H15, E5).**
- Before, every box was a fixed 0.45 × 0.45 × 0.50 m, with its bottom at the measured point.
- Now `geometry.measured_size` gives width = box width × range ÷ fx and height = box height × range ÷ fy. The range is the depth reading, or the lidar cluster's distance along the camera axis.
- The camera cannot see an object's depth, so depth is assumed equal to the width.
- The marker cube and the `Detection3D` box are centred on the measured point. A track's size is that of its latest matched detection, not smoothed.
- When the size is unknown, `Track.size` is zero and the boxes fall back to the old fixed size.
- The run log's track entries gain a `size` key.

**Diagnostics (H14, E4).**
- All existing keys keep their values. New keys:
  - `throughput_hz`: frames finished in the last 5 s of wall clock, divided by 5. It falls to 0 when frames stop.
  - `rgb_range_gap_ms`: |colour image stamp − depth image or scan stamp| of the last processed frame.
  - `tentative_tracks`, `confirmed_tracks`, `coasting_tracks` and `dynamic_tracks`.
  - `dropped_frames`, `tf_failures`, `images_received` and `range_msgs_received`.
- **WARN rule:** latency over 100 ms, or `throughput_hz` under 2 Hz (the p95 rate gate) once frames have arrived for 5 s. This replaces `fps < 10`, which measured how fast one frame is processed, not how often frames are processed. The 5 s wait stops a false WARN while the first window fills.

**Package.** `r_one` now has `exec_depend` on `r_one_msgs`.

## Costmap clearing (E1)

The master plan said "publish an empty cloud so costmaps clear". That holds for a consumer that replaces its state with each message, but not for Nav2's `nav2_costmap_2d::ObstacleLayer`. In Nav2 Humble's [obstacle_layer.cpp](https://github.com/ros-navigation/navigation2/blob/humble/nav2_costmap_2d/plugins/obstacle_layer.cpp):

- `updateBounds` writes `LETHAL_OBSTACLE` into the layer's own grid for each marking point, and never resets the grid.
- A mark is removed only by `raytraceFreespace` from a clearing source, by footprint clearing, or by `reset()`.
- An empty marking cloud runs its marking loop zero times, so it removes nothing.

Setting `clearing: True` on the `dynamic_obstacles` source would not help either. The cloud is in `odom`, so its rays would start at the odom origin, not at a sensor.

So:
- The outputs are published every frame, which is correct for any consumer that reads the latest message, including `perception/tracks`.
- With today's `ObstacleLayer` setup, a predicted-path mark stays until the laser scan raytraces through it.
- The gate "costmap clears ≤ 1 s after a track ends" moves to stage 4c. There, our own costmap layer repaints from `perception/tracks` on every update. Nav2 is not launched until stage 3b, so this could not have been measured live in 2e anyway.

## Tests

New unit tests cover:
- the lifecycle states, and that an ended track is returned by nothing ([test_tracking.py](../r_one/test/test_tracking.py));
- `measured_size`, the size-aware markers and `Detection3D` boxes, and every `Track` field, including a row-major check with a non-symmetric covariance ([test_modules.py](../r_one/test/test_modules.py));
- empty outputs on an empty frame, a walker published as one track with its size, nothing published before the intrinsics arrive, and each new diagnostics key and WARN rule ([test_spatial_vision_guards.py](../r_one/test/test_spatial_vision_guards.py)).

## Results (2026-11-18)

**Container tests:** 225 tests, 0 errors, 0 failures, 0 skipped. That is 203 at 2c plus 22 new: guard tests 41 → 50, module tests 24 → 33 and tracking tests 28 → 32.

**Bag replay**, depth-image mode, with the command from [baseline_2a.md](baseline_2a.md) plus `-e ROS_LOCALHOST_ONLY=1`. One run, against the three 2c live runs, which were identical:

| Metric | 2c live | 2e | Gate |
|---|---|---|---|
| Images received (of 232) | 231 | 228 | |
| Frames | 115 | 113 | |
| Walker ID switches | 0 | 0 | ≤ 1 |
| Time to DYNAMIC (s) | 0.60 | 0.80 | ≤ 1.5 |
| Rate, p95 (Hz) | 4.98 | 4.98 | ≥ 2 |
| Walker recall | 0.991 | 0.991 | ≥ 0.90 |
| Position RMSE (m) | 0.216 | 0.215 | ≤ 0.25 |
| Velocity RMSE, cruising (m/s) | 0.163 | 0.138 | ≤ 0.25 |
| False DYNAMIC rate | 0 | 0 | ≤ 1% |
| Ghosts | 0 | 0 | 0 |
| Latency, p95 (s) | 0.030–0.126 | 0.029 | |

- The 2e run is missing the 2c runs' first two frames (29.67 s and 29.87 s), and every other frame stamp is the same. That start accounts for the later DYNAMIC time and the lower cruise velocity error. 2e does not change tracking in depth mode. Why the node subscribed later was not measured. That run also had two extra `ros2 topic echo` subscribers.
- Overall velocity RMSE is 0.335 (2c: 0.339), which `score_run.py --gates` marks as failed, for the reason in [timing_2c.md](timing_2c.md#results).
- `perception/tracks` during the replay showed the walker as `CONFIRMED` and `DYNAMIC`, measured at 0.74 × 0.74 × 1.60 m, and the standing person as `CONFIRMED` and `STATIC`, measured at 0.57 × 0.57 × 1.64 m. A 0.26-score `skis` detection appeared as `TENTATIVE`. That is why consumers should filter by state.
- `/diagnostics` was OK, with `throughput_hz` 5.00, `rgb_range_gap_ms` 0.00 (the bag's colour and depth stamps are identical), 0 dropped frames and 0 TF failures. `fps` read 39.91: that measures how fast one frame is processed, not the rate.

## Known limits

- **Box size jitters** with the detection box from frame to frame, because it is not smoothed. Smoothing would be a tuning choice for later.
- **Depth is assumed equal to width.** A long object seen end-on, such as a bench, gets too small a depth.
- **In lidar mode the box is centred on the scan plane.** The measured point is where the beams hit, at the lidar's height, so a tall object's box can extend below the floor. The demo uses the depth image, whose point is at the box centre. Lidar mode is still unproven on people in simulation (C1).
- **Nav2's `ObstacleLayer` does not clear on empty clouds** (see above). Stage 4c's layer is the fix.
