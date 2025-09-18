# droneswarm/droneflight_single_realistic.py
# Single Wilmington flight with hard negatives + spoof onset after 10 minutes
# Generates overlay, ENU/zoom plots, and wide CSVs (imu_truth.csv, imu_spoof.csv)

import math, os, io, urllib.parse, urllib.request, re, csv
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# =========================
# CONFIG
# =========================
API_KEY = "AIzaSyBviNwbX4Ktsi6_6wVOIYc9K1mcLHIfWS8"

LAUNCH_DMS       = ("34°13'31.1\"N", "77°46'22.9\"W")
TRUTH_TARGET_DMS = ("34°13'33.7\"N", "77°52'18.8\"W")
DRIFT_TARGET_DMS = ("34°13'28.7\"N", "77°52'18.2\"W")

N_DRONES       = 10
DRONE_SPREAD_M = 80.0
FLIGHT_MIN     = 30
FIX_RATE_HZ    = 1
N_STEPS        = FLIGHT_MIN * 60 * FIX_RATE_HZ

SPOOF_START_MIN = 10
SPOOF_START_STEP = SPOOF_START_MIN * 60 * FIX_RATE_HZ

# Spoof target offset (meters)
SPOOF_FINAL_DRIFT_M = 200.0

# Satellite overlay image
IMG_SIZE = (1280, 1024)
MAPTYPE  = "satellite"
STATIC_SCALE = 2
LINESTYLE = "--"
LW        = 1.2

# =========================
# GEO HELPERS
# =========================
def dms_to_dd(dms_str: str) -> float:
    s = dms_str.strip().replace(" ", "")
    m = re.match(r'(\d+)[°](\d+)[\']([\d\.]+)["]([NSEW])', s, re.IGNORECASE)
    deg, mins, secs, hemi = float(m[1]), float(m[2]), float(m[3]), m[4].upper()
    dd = deg + mins/60.0 + secs/3600.0
    return -dd if hemi in ["S","W"] else dd

def meters_per_deg(lat_deg: float):
    lat_m = 111_132.0
    lon_m = 111_320.0 * math.cos(math.radians(lat_deg))
    return lat_m, lon_m

# =========================
# HARD NEGATIVES
# =========================
def apply_hard_negatives(lats, lons, rng):
    # Wind bias (AR1)
    wind = rng.normal(0, 0.05, size=len(lats)).cumsum() * 1e-6
    lats = lats + wind

    # GPS jitter
    jitter_idx = rng.choice(len(lats), size=20, replace=False)
    lats[jitter_idx] += rng.normal(0, 3e-6, size=len(jitter_idx))
    lons[jitter_idx] += rng.normal(0, 3e-6, size=len(jitter_idx))

    return lats, lons

# =========================
# TRACK BUILDER
# =========================
def build_tracks(latA, lonA, latT, lonT, latS, lonS,
                 n_drones, n_steps, spread_m, spoof_start_step):
    truth_lat_base = np.linspace(latA, latT, n_steps)
    truth_lon_base = np.linspace(lonA, lonT, n_steps)

    spoof_lat_base = truth_lat_base.copy()
    spoof_lon_base = truth_lon_base.copy()
    for i in range(spoof_start_step, n_steps):
        frac = (i - spoof_start_step) / (n_steps - spoof_start_step)
        spoof_lat_base[i] = (1-frac)*truth_lat_base[i] + frac*latS
        spoof_lon_base[i] = (1-frac)*truth_lon_base[i] + frac*lonS

    lat_scale, _ = meters_per_deg(latA)
    offsets = np.linspace(-spread_m/2, spread_m/2, n_drones)

    truth_lats, truth_lons, spoof_lats, spoof_lons = [], [], [], []
    rng = np.random.default_rng(1234)

    for off in offsets:
        tlats = truth_lat_base + off/lat_scale
        tlons = truth_lon_base.copy()
        slats = spoof_lat_base + off/lat_scale
        slons = spoof_lon_base.copy()

        # Hard negatives
        tlats, tlons = apply_hard_negatives(tlats, tlons, rng)
        slats, slons = apply_hard_negatives(slats, slons, rng)

        truth_lats.append(tlats)
        truth_lons.append(tlons)
        spoof_lats.append(slats)
        spoof_lons.append(slons)

    return truth_lons, truth_lats, spoof_lons, spoof_lats

# =========================
# IMU SYNTH
# =========================
def synthesize_imu(lats, lons, seed):
    rng = np.random.default_rng(seed)
    lat0, lon0 = lats[0], lons[0]
    lat_m, lon_m = meters_per_deg(lat0)
    E = (np.array(lons) - lon0) * lon_m
    N = (np.array(lats) - lat0) * lat_m
    t = np.arange(len(lats))

    vE = np.gradient(E, t)
    vN = np.gradient(N, t)
    aE = np.gradient(vE, t)
    aN = np.gradient(vN, t)

    # IMU noise: bias + drift + white
    acc_biasE = rng.normal(0, 0.02)
    acc_biasN = rng.normal(0, 0.02)
    acc_rwE = rng.normal(0, 0.001, len(t)).cumsum()*1e-3
    acc_rwN = rng.normal(0, 0.001, len(t)).cumsum()*1e-3
    accE = aE + acc_biasE + acc_rwE + rng.normal(0, 0.05, len(t))
    accN = aN + acc_biasN + acc_rwN + rng.normal(0, 0.05, len(t))

    gyroX = rng.normal(0, 0.05, len(t))
    gyroY = rng.normal(0, 0.05, len(t))

    return accE, accN, gyroX, gyroY

# =========================
# GOOGLE STATIC MAP HELPERS
# =========================
TILE_SIZE = 256
GOOGLE_STATIC_MAX = 640
def lonlat_to_pixel(lon, lat, zoom, tile_size=TILE_SIZE):
    siny = max(min(math.sin(math.radians(lat)),0.9999),-0.9999)
    x = tile_size*(0.5+lon/360.0)*(2**zoom)
    y = tile_size*(0.5-0.5*math.log((1+siny)/(1-siny))/math.pi)*(2**zoom)
    return x,y
def fit_zoom_for_bounds(lon_min,lat_min,lon_max,lat_max,width,height,padding=60,max_zoom=20):
    for z in range(max_zoom,-1,-1):
        x_min,y_max = lonlat_to_pixel(lon_min,lat_min,z)
        x_max,y_min = lonlat_to_pixel(lon_max,lat_max,z)
        if (abs(x_max-x_min)+2*padding)<=width and (abs(y_max-y_min)+2*padding)<=height:
            return z,(lon_min+lon_max)/2,(lat_min+lat_max)/2
    return 0,(lon_min+lon_max)/2,(lat_min+lat_max)/2
def fetch_google_static(center_lon,center_lat,zoom,size,maptype,api_key,scale=2):
    req_w=min(GOOGLE_STATIC_MAX,max(1,size[0]//scale))
    req_h=min(GOOGLE_STATIC_MAX,max(1,size[1]//scale))
    params={"center":f"{center_lat},{center_lon}","zoom":str(zoom),
            "size":f"{req_w}x{req_h}","scale":str(scale),
            "maptype":maptype,"key":api_key}
    url="https://maps.googleapis.com/maps/api/staticmap?"+urllib.parse.urlencode(params)
    with urllib.request.urlopen(url) as resp:
        data=resp.read()
    img=Image.open(io.BytesIO(data)).convert("RGBA")
    if img.size!=size: img=img.resize(size,Image.LANCZOS)
    return img
def rgba_from_tab10(i):
    r,g,b,_=cm.get_cmap("tab10")(i%10)
    return (int(r*255),int(g*255),int(b*255),230)

# =========================
# DRAW
# =========================
def draw_dashed_polyline(draw, pts, dash_len=8, gap_len=6, width=3, color=(255,255,255,255)):
    from math import hypot
    for (x1,y1),(x2,y2) in zip(pts[:-1],pts[1:]):
        dx,dy=x2-x1,y2-y1
        seg_len=hypot(dx,dy)
        if seg_len==0: continue
        ux,uy=dx/seg_len,dy/seg_len
        dist,draw_dash=0.0,True
        while dist<seg_len:
            d=min(dash_len if draw_dash else gap_len,seg_len-dist)
            if draw_dash:
                xa,ya=x1+ux*dist,y1+uy*dist
                xb,yb=x1+ux*(dist+d),y1+uy*(dist+d)
                draw.line([(xa,ya),(xb,yb)],fill=color,width=width)
            dist+=d; draw_dash=not draw_dash

def draw_tracks_on_image(img,center_lon,center_lat,zoom,lon_tracks,lat_tracks,width=3):
    W,H=img.size
    cx,cy=lonlat_to_pixel(center_lon,center_lat,zoom)
    draw=ImageDraw.Draw(img,"RGBA")
    for idx,(lon,lat) in enumerate(zip(lon_tracks,lat_tracks)):
        pts=[(lonlat_to_pixel(lo,la,zoom)[0]-cx+W/2,
              lonlat_to_pixel(lo,la,zoom)[1]-cy+H/2) for lo,la in zip(lon,lat)]
        col=rgba_from_tab10(idx)
        draw_dashed_polyline(draw,pts,width=width,color=col)

# =========================
# MAIN
# =========================
def main():
    latA=dms_to_dd(LAUNCH_DMS[0]); lonA=dms_to_dd(LAUNCH_DMS[1])
    latT=dms_to_dd(TRUTH_TARGET_DMS[0]); lonT=dms_to_dd(TRUTH_TARGET_DMS[1])
    latS=dms_to_dd(DRIFT_TARGET_DMS[0]); lonS=dms_to_dd(DRIFT_TARGET_DMS[1])

    truth_lons,truth_lats,spoof_lons,spoof_lats=build_tracks(
        latA,lonA,latT,lonT,latS,lonS,
        n_drones=N_DRONES,n_steps=N_STEPS,spread_m=DRONE_SPREAD_M,
        spoof_start_step=SPOOF_START_STEP
    )

    all_lons=np.concatenate([*truth_lons,*spoof_lons])
    all_lats=np.concatenate([*truth_lats,*spoof_lats])
    lon_min,lon_max=float(all_lons.min()),float(all_lons.max())
    lat_min,lat_max=float(all_lats.min()),float(all_lats.max())
    zoom,center_lon,center_lat=fit_zoom_for_bounds(lon_min,lat_min,lon_max,lat_max,
        IMG_SIZE[0],IMG_SIZE[1],padding=60,max_zoom=20)
    img=fetch_google_static(center_lon,center_lat,zoom,IMG_SIZE,MAPTYPE,API_KEY,scale=STATIC_SCALE)

    draw_tracks_on_image(img,center_lon,center_lat,zoom,truth_lons,truth_lats,width=3)
    draw_tracks_on_image(img,center_lon,center_lat,zoom,spoof_lons,spoof_lats,width=3)
    img.save("swarm_google_satellite_overlay.png")
    print("Saved swarm_google_satellite_overlay.png")

    # CSV wide logs
    timestamps=np.arange(N_STEPS)
    rows_truth,rows_spoof=[],[]
    for i,t in enumerate(timestamps):
        rowT={"timestamp_sec":int(t)}
        rowS={"timestamp_sec":int(t)}
        for d in range(N_DRONES):
            accE_T,accN_T,gyroX_T,gyroY_T=synthesize_imu(truth_lats[d],truth_lons[d],seed=42+d)
            accE_S,accN_S,gyroX_S,gyroY_S=synthesize_imu(spoof_lats[d],spoof_lons[d],seed=42+d)
            rowT.update({
                f"drone{d+1}_lat":truth_lats[d][i],
                f"drone{d+1}_lon":truth_lons[d][i],
                f"drone{d+1}_IMUx":accE_T[i],
                f"drone{d+1}_IMUy":accN_T[i],
                f"drone{d+1}_GRUx":gyroX_T[i],
                f"drone{d+1}_GRUy":gyroY_T[i],
            })
            rowS.update({
                f"drone{d+1}_lat":spoof_lats[d][i],
                f"drone{d+1}_lon":spoof_lons[d][i],
                f"drone{d+1}_IMUx":accE_S[i],
                f"drone{d+1}_IMUy":accN_S[i],
                f"drone{d+1}_GRUx":gyroX_S[i],
                f"drone{d+1}_GRUy":gyroY_S[i],
            })
        rows_truth.append(rowT); rows_spoof.append(rowS)

    pd.DataFrame(rows_truth).to_csv("imu_truth.csv",index=False)
    pd.DataFrame(rows_spoof).to_csv("imu_spoof.csv",index=False)
    print("Saved imu_truth.csv and imu_spoof.csv")

if __name__=="__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
