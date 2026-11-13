# One tracked object. Coordinates use the frame in TrackArray.header.frame_id.

# Track lifecycle
uint8 STATE_TENTATIVE=0  # Seen, but not yet confirmed by enough detections
uint8 STATE_CONFIRMED=1  # Matched to a detection in this frame
uint8 STATE_COASTING=2   # Not detected in this frame, position is predicted

# Motion decision
uint8 MOTION_UNKNOWN=0
uint8 MOTION_STATIC=1
uint8 MOTION_DYNAMIC=2

uint32 id                    # Stable for the life of the track
string class_name            # Detector class, such as "person"
float32 score                # Detector confidence of the latest match, 0 to 1
uint8 state                  # One of STATE_*
uint8 motion                 # One of MOTION_*
geometry_msgs/Point position        # Metres
geometry_msgs/Vector3 velocity      # Metres per second
geometry_msgs/Vector3 size          # Metres, zero when unknown
float64[16] covariance       # Row-major 4x4 covariance of [x, y, vx, vy]
float32 age_s                # Seconds since the track was created
float32 time_since_update_s  # Seconds since the last matched detection
