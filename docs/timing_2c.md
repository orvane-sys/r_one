# Stage 2c: timing and ranging

Stage 2c moved inference off the ROS executor, fixed the stamp and transform used to place each detection, and replaced the lidar and depth ranging. This page records what changed, how it scores on the sample bag, and what is still unproven.

**Result:**
- All 203 tests pass in the container.
- With every image delivered, the node processes the bag at 5 Hz (the bag's 10 Hz with `process_every_n_frames: 2`). Walker recall rose from 0.896 to 0.990, with 0 ID switches and 0 false DYNAMIC.
- Fast DDS's default shared memory was dropping 27–58% of the camera images before they reached any subscriber. The Docker image now loads a profile with a larger segment (see [Image delivery](#image-delivery)).
- Lidar ranging is proven on synthetic scans only. The simulated lidar cannot see the people in the demo (C1).
- The 3-run live gate is in [Live gate](#live-gate).

## What changed

**Worker thread (H10).**
- The sync callbacks put each frame set into a one-slot buffer, [frame_slot.py](../r_one/r_one/frame_slot.py), which keeps only the newest set.
- One worker thread takes it and runs detection, ranging, TF, tracking and publishing. A frame replaced before the worker took it is counted in `dropped_frames`.
- The worker warms the detector up on a black frame first. Before, the first real frame took 5.3 s.
- Only tracking, the world model, publishing and the run log hold the state lock. The diagnostics timer and the services no longer wait for inference. `TestSlowDetector` checks this with a detector that blocks for 5 s.

**Detector.**
- `UltralyticsDetector.detect` now calls `predict()`, not `track()` with ByteTrack.
- ByteTrack hid a new track's box for that track's first frame. On the step 2 run, the 11 frames it hid were exactly the 11 frames with no walker box.
- The node never used ByteTrack's IDs, because our tracker assigns its own.
- `predict()` filters at confidence 0.25, where `track()` used 0.1 (ultralytics 8.4.37, `model.py:576`). On the bag the lowest person score was 0.748, so this made no difference there. A `conf` parameter belongs to stage 4a.
- `lapx` was only needed by ByteTrack, so it is removed from [requirements.txt](../requirements.txt).

**Stamps and TF (H8, D3).**
- Both modes now stamp a frame with the colour image's stamp, because the boxes come from that image. Before, lidar mode used the scan's stamp and depth mode used the depth image's.
- A detection is moved to `target_frame` with the transform at that stamp, waiting up to 0.1 s. If the lookup fails, the detection is dropped and counted in `tf_failures`.
- Before, a failed lookup silently fell back to the latest transform. In the first measured run, 142 of 148 measurements fell back. After the worker change, none needed it.

**Lidar ranging (H16, H1, D4).**
- **Before:** the box centre's camera bearing was reused as a lidar bearing, and the median of 5 beams was taken. This ignored the 0.205 m between the lidar and the camera. A box centre that fell between a person's legs also hit the wall behind them.
- **Now** ([geometry.py](../r_one/r_one/geometry.py)):
  1. `scan_points` turns the valid beams into points.
  2. They are moved into the camera frame with the static TF. The node looks this transform up once and caches it.
  3. `points_in_box` keeps the points that project into the box.
  4. `nearest_cluster` splits them at range jumps over 0.3 m, and returns the centre of the nearest group with at least 2 beams.
  5. That centre is placed from the camera frame.

| Synthetic scene | Old code | New code |
|---|---|---|
| Two legs at 1.0 m, 0.2 m apart, wall at 4.0 m | 4.0 m | 0.954 m |
| A post 1.5 m ahead, 0.6 m to the side | 4.0 m | 1.459 m |

**Depth ranging (M12).**
- `mono16` depth is read as millimetres, like `16UC1`.
- A depth image of a different size from the colour image skips the frame, with a throttled error. Ranging reads depth at the box's pixels, so the two images must be registered.
- The depth limits are now the named constants `DEPTH_MIN_M` and `DEPTH_MAX_M`, with unchanged values (0.1 and 10.0 m).
- The range is still the 25th percentile of a patch over the middle third of the box.

**Debug window (M8).**
- Every OpenCV window call is made on the worker thread. The worker opens the window once and closes it when it stops.
- `main()` stops the node before its `destroyAllWindows` backstop.
- Turning `show_debug_window` off at runtime now closes the window. Before, it stayed open and frozen.

**Run log (D2).**
- Each frame gains the keys `detections` (class, score, box and whether it was ranged), `dropped_frames`, `tf_failures`, `images_received` and `range_msgs_received`.
- Nothing is removed or renamed.

**Fast DDS profile.** The Docker image sets `FASTRTPS_DEFAULT_PROFILES_FILE` to [docker/fastdds_profile.xml](../docker/fastdds_profile.xml). See [Image delivery](#image-delivery).

No ROS parameter, topic, message or tuned tracker value changed.

## Decisions

- **D1. The time-to-DYNAMIC target needs 4 Hz.** Stage 2d carried a gate that the 2 Hz walker sweep must meet 1.5 s. With the current tracker it can't, unless the false-DYNAMIC and ID-switch gates break (see [the sweep](#walker-sweep)). The decision:
  - The 1.5 s target stays.
  - The sweep test asserts it at 4 Hz and 5 Hz.
  - 2 Hz keeps its 1.8 s regression bound.
  - The live gate requires a median processed rate of at least 4 Hz.
- **D2. The run log only gains keys.** `score_run.py` ignores keys it doesn't know.
- **D3. No fallback to the latest transform.** A measurement whose transform fails at its own stamp is dropped and counted, not placed with the wrong pose.
- **D4. `extract_lidar_distance` is replaced** by beam projection and clustering. It was internal to the package.

## How it was measured

- **Bag replay.** The sample bag is played through the full node in the Docker image in depth-image mode. The command is the one in [baseline_2a.md](baseline_2a.md), plus `-e ROS_LOCALHOST_ONLY=1`. Each run is scored with [score_run.py](../tools/score_run.py). The median rate and the largest gap are computed from the frame stamps in the run log.
- **The bag.** Its colour and depth images have identical stamps, the same 640×480 size, and `32FC1` depth. So the stamp and depth-format fixes do not change a depth-mode replay. Only the worker, the warm-up and the detector switch can.
- **Lidar.** The bag's scan reads `inf` across the whole camera view, so a lidar-mode replay only shows that the code runs. It ranged 0 of 222 detections, with no errors.

## Results

One bag replay per step:

| Metric | Before (step 1) | Worker (step 2) | `predict()` (step 3) | Ranging (steps 4–5) | Gate |
|---|---|---|---|---|---|
| Frames | 77 | 65 | 97 | 96 | |
| Walker ID switches | 0 | 3 ✗ | 0 | 0 | ≤ 1 |
| Walker recall | 0.896 ✗ | 0.785 ✗ | 0.990 | 0.990 | ≥ 0.90 |
| Position RMSE (m) | 0.219 | 0.216 | 0.214 | 0.217 | ≤ 0.25 |
| Velocity RMSE, cruising (m/s) | 0.144 | 0.246 | 0.142 | 0.161 | ≤ 0.25 |
| Velocity RMSE, overall (m/s) | 0.233 | 0.418 | 0.301 | 0.353 | see below |
| False DYNAMIC rate | 0 | 0.143 ✗ | 0 | 0 | ≤ 1% |
| Time to DYNAMIC (s) | 0.86 | 0.86 | 0.60 | 0.60 | ≤ 1.5 |
| Ghosts | 0 | 0 | 0 | 0 | 0 |
| Rate, p95 (Hz) | 2.49 | 0.91 ✗ | 2.00 | 2.50 | ≥ 2 |
| Rate, median (Hz) | 5.00 | 4.98 | 5.00 | 5.00 | ≥ 4 (D1) |
| Largest gap between frames (s) | 5.3 | 1.3 | 0.6 | 0.9 | |
| Latency, p95 (s) | 0.235 | 0.124 | 0.085 | 0.086 | |

Notes on the table:
- **Step 1's 5.3 s gap** was the first inference. The warm-up removed it.
- **Step 2's misses** were mostly ByteTrack's hidden boxes (see [What changed](#what-changed)). Two hidden walker boxes in a row came just before an ID swap, which left the standing person DYNAMIC for 9 frames.
- **Velocity.** [tracker_2d.md](tracker_2d.md) defines cruising velocity as the live gate (≤ 0.25), and every run passes it. Overall velocity can't reach 0.25 on this script, because the walker reverses instantly. Its ≤ 0.35 bound is checked on the fixture, where it passes. `score_run.py --gates` still checks overall velocity against 0.25, so it reports that line as failed on every run.
- **Steps 4–5 vs step 3.** The overall velocity rose from 0.301 to 0.353. The changes in steps 4–5 cannot reach a depth-mode replay of this bag (see [How it was measured](#how-it-was-measured)). A different set of frames reached the node (207 images received against 201). With one run per step, the spread between runs is not measured.
- **Transport.** These runs used the default Fast DDS transport, which loses images (see [Image delivery](#image-delivery)). They reached 201–207 of 232 images. Later runs on the same setup reached as few as 97.

## Walker sweep

`test_walker_first_seen_anywhere_on_its_path` in [test_tracking.py](../r_one/test/test_tracking.py) starts a track every 0.1 s along the walker's 9 s path, with 0.1 m noise, and measures the time to DYNAMIC. The test uses seeds 0–1, and the wider sweep uses seeds 0–19 (1800 starts per rate).

| Rate | Worst, seeds 0–1 | Worst, seeds 0–19 | Starts over 1.5 s | Test bound |
|---|---|---|---|---|
| 5 Hz | 1.3 s | 1.5 s | 0 | 1.5 s |
| 4 Hz | 1.4 s | 1.6 s | 6 | 1.5 s |
| 3 Hz | 1.7 s | 1.8 s | 24 | |
| 2 Hz | 1.7 s | 2.2 s | 96 | 1.8 s |

The 4 Hz misses all start near a turn (at 3.3, 7.6 and 7.8 s). The 4 Hz test passes with its two seeds, so it guards against regressions; it does not prove that every start meets 1.5 s.

## Image delivery

The first four attempts at the live gate failed. Across them, the node received only 103–169 of the bag's 232 colour images, and 118–167 even on an idle machine.
- The node itself kept up: about 25 ms per frame, with no dropped frames.
- The gaps fell at different bag times in each run.
- In the three idle-machine runs, every ID switch, late DYNAMIC and false DYNAMIC came after a gap of 1 s or more.

**Host test.** Fast DDS 2.6.12 sent 100 frames at 10 Hz, at the bag's sizes (0.92 MB colour, 1.23 MB depth), to a subscriber with the node's QoS (best effort, depth 5):

| Transport | Colour received | Depth received |
|---|---|---|
| Default, `ROS_LOCALHOST_ONLY=1` | 3 | 0 |
| UDP only | 2 | 1 |
| Cyclone DDS | 3 | 1 |
| Default, but a RELIABLE subscriber | 100 | 100 |
| Shared memory with a 4 MB segment, plus loopback UDP (two runs) | 100 | 100 |
| Same with 10×10 px images and the default transport | 100 | 100 |

**Container replay.** The full node ran with a second, plain subscriber that counted the images:

| Transport | Counter (colour / depth) | Node images | Median rate | p95 rate | ID switches | Time to DYNAMIC |
|---|---|---|---|---|---|---|
| Default | 62 / 80 | 97 | 2.2 Hz | 0.4 Hz | 3 | 2.4 s |
| 16 MB shared-memory segment | 232 / 231 | 231 | 5.0 Hz | 5.0 Hz | 0 | 0.6 s |

So the images are lost in transport, before any subscriber sees them. Fast DDS's default shared-memory segment is 512 KB, smaller than either image. That the burst overruns it is our reading; the fix is measured.

**The fix.** [docker/fastdds_profile.xml](../docker/fastdds_profile.xml) gives shared memory a 4 MB segment, and limits UDP to `127.0.0.1`. The Dockerfile sets `FASTRTPS_DEFAULT_PROFILES_FILE` to it.
- A RELIABLE subscriber also fixed delivery. It was not chosen, because a RELIABLE subscriber receives nothing from a best-effort publisher, which many camera drivers are.
- Humble's `rmw_fastrtps` adds its own UDP and shared-memory transports when `ROS_LOCALHOST_ONLY=1` ([participant.cpp](https://github.com/ros2/rmw_fastrtps/blob/humble/rmw_fastrtps_shared_cpp/src/participant.cpp)). The profile was tested with and without that variable.

## Live gate

The gate runs three bag replays. On every run:
- ID switches ≤ 1;
- time to DYNAMIC ≤ 1.5 s;
- median rate ≥ 4 Hz;
- p95 rate ≥ 2 Hz.

Three replays with the [Fast DDS profile](#image-delivery) (4 MB segment), on 2026-11-17:

| Metric | Run 1 | Run 2 | Run 3 | Gate |
|---|---|---|---|---|
| Images received (of 232) | 231 | 231 | 231 | |
| Frames | 115 | 115 | 115 | |
| Walker ID switches | 0 | 0 | 0 | ≤ 1 |
| Time to DYNAMIC (s) | 0.60 | 0.60 | 0.60 | ≤ 1.5 |
| Rate, median (Hz) | 5.00 | 5.00 | 5.00 | ≥ 4 |
| Rate, p95 (Hz) | 4.98 | 4.98 | 4.98 | ≥ 2 |
| Largest gap between frames (s) | 0.20 | 0.20 | 0.20 | |
| Walker recall | 0.991 | 0.991 | 0.991 | ≥ 0.90 |
| Position RMSE (m) | 0.216 | 0.216 | 0.216 | ≤ 0.25 |
| Velocity RMSE, cruising (m/s) | 0.163 | 0.163 | 0.163 | ≤ 0.25 |
| False DYNAMIC rate | 0 | 0 | 0 | ≤ 1% |
| Ghosts | 0 | 0 | 0 | 0 |
| Latency, p95 (s) | 0.030 | 0.126 | 0.121 | |

**The gate passes on all three runs.**
- With every image delivered, the three runs processed the same 115 frames, so the tracking scores are identical. Only the latency differs.
- Overall velocity RMSE was 0.339 on each run. `score_run.py --gates` marks it as failed, for the reason in [Results](#results).
- No frames were dropped and no TF lookups failed.

## Known limits

- **Lidar ranging is proven on synthetic scans only.** Gazebo Classic's lidar cannot see actors (C1). Stage 3a checks whether Harmonic's `gpu_lidar` can.
- **The laser-to-camera transform is cached for the life of the node.** If the extrinsics are republished, restart the node.
- **Depth images of a different size are skipped, not registered.**
- **The Fast DDS profile keeps ROS traffic on this machine.** Nodes on another computer cannot see the container's topics. Native installs do not get the profile unless they export `FASTRTPS_DEFAULT_PROFILES_FILE` (see the README).
- **The live gate used one bag and one machine.** With every image delivered, repeat replays process the same frames. So three passing runs show the transport is stable, not that the tracker copes with other scenes.
- **Detections scoring between 0.1 and 0.25 are no longer tracked.** This follows from `predict()`'s default.
- **Each step was replayed once,** so differences smaller than the spread between runs are not proven.
- **`main()`'s shutdown order has no test,** because `main()` is not unit-tested.
