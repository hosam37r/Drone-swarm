# DroneSwarm – Single-Flight GPS Spoofing Simulator (Wilmington, NC)

This script (`droneswarm/droneflight_single_realistic.py`) simulates a 30-minute drone swarm mission over Wilmington, NC and injects a **slow-drift GPS spoofing** attack beginning at the 10-minute mark. It renders a satellite **overlay image** of truth and spoofed trajectories and exports two “wide” CSV logs with per-drone IMU and gyro signals for downstream analysis.

---

## Features

- **Swarm trajectory generation** from launch point to a truth target
- **Spoofed target blending** after 10 minutes, steering GPS toward a false endpoint
- **Hard negatives** for realism:
  - Wind bias (random walk)
  - GPS jitter bursts
  - Formation spread
  - Noisy IMU/gyro with bias + drift + white noise
- **Satellite overlay image** with dashed truth vs spoofed trajectories
- **Per-second CSV logs** for truth and spoof cases:
  - `imu_truth.csv`
  - `imu_spoof.csv`

---

## Requirements

- Python 3.9+
- Dependencies:
  ```bash
  pip install numpy pillow matplotlib pandas
  ```
- Google Static Maps API key (with billing enabled)

---

## Usage

From repository root:

```bash
python droneswarm/droneflight_single_realistic.py
```

Outputs:

- `swarm_google_satellite_overlay.png`
- `imu_truth.csv`
- `imu_spoof.csv`

---

## Configuration

Edit the **CONFIG** block in the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `API_KEY` | `""` | Google Static Maps key |
| `LAUNCH_DMS` | `("34°13'31.1\"N","77°46'22.9\"W")` | Launch coordinates |
| `TRUTH_TARGET_DMS` | `("34°13'33.7\"N","77°52'18.8\"W")` | Intended mission target |
| `DRIFT_TARGET_DMS` | `("34°13'28.7\"N","77°52'18.2\"W")` | Spoofed target |
| `N_DRONES` | `10` | Number of drones in swarm |
| `DRONE_SPREAD_M` | `80.0` | Formation spread (m) |
| `FLIGHT_MIN` | `30` | Flight duration (minutes) |
| `FIX_RATE_HZ` | `1` | Sampling rate (Hz) |
| `SPOOF_START_MIN` | `10` | Spoof onset (minutes) |
| `SPOOF_FINAL_DRIFT_M` | `200.0` | Spoof offset (m) |
| `IMG_SIZE` | `(1280,1024)` | Overlay size (px) |
| `MAPTYPE` | `"satellite"` | Map style |
| `STATIC_SCALE` | `2` | Google map scale |

---

## Outputs

### 1. Satellite overlay
- `swarm_google_satellite_overlay.png`
- Truth (dashed polylines) and spoofed tracks rendered on Google satellite map

### 2. CSV logs
- `imu_truth.csv` and `imu_spoof.csv`
- Schema (wide format):

| Column | Description |
|--------|-------------|
| `timestamp_sec` | Time in seconds |
| `drone{k}_lat`, `drone{k}_lon` | Latitude/Longitude |
| `drone{k}_IMUx`, `drone{k}_IMUy` | Accelerations (m/s²) |
| `drone{k}_GRUx`, `drone{k}_GRUy` | Gyro roll/pitch rates (rad/s) |

---

## Notes

- ENU/zoom plots mentioned in the header comment are **not implemented**; only the satellite overlay and CSVs are produced.
- The geodetic model is simplified; suitable for visualization/ML experiments, not for navigation-grade accuracy.
- Ensure your **API key is set securely** (via environment variable rather than in-file constant).

---

## Example Environment Setup

```bash
export GOOGLE_MAPS_API_KEY="YOUR_KEY_HERE"
python droneswarm/droneflight_single_realistic.py
```

---

## File Layout

```
droneswarm/
└─ droneflight_single_realistic.py
swarm_google_satellite_overlay.png   # created on run
imu_truth.csv                        # created on run
imu_spoof.csv                        # created on run
```

---

## Troubleshooting

- **Blank/failed map** → check API key and billing
- **No tracks drawn** → ensure map size and zoom bounds cover trajectory
- **Wide CSV** → use `pandas` or a data viewer; Excel may wrap columns

---

## License & Attribution

- Respect Google Static Maps API terms of use
- Replace placeholder key with your own
- Do not commit real API keys to version control
