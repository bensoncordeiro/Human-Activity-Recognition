import os
import pandas as pd
import numpy as np


RAW_ROOT = "zips"

# Output folder for merged, trimmed sessions
PROCESSED_SESSION_DIR = "DATA"
os.makedirs(PROCESSED_SESSION_DIR, exist_ok=True)

# Target sampling frequency = 100 Hz
TARGET_FS = 100                         # Hz
TARGET_DT_MS = int(1000 / TARGET_FS)    # 10 ms bins

#Helper functions
def load_sensor_csv(path, prefix):
    """
    Load a sensor CSV with columns:
        time, seconds_elapsed, z, y, x

    Converts 'time' (nanoseconds since epoch) to a DatetimeIndex and
    renames axes to: prefix_x, prefix_y, prefix_z.
    """
    df = pd.read_csv(path)

    # converting nanosecond epoch to datetime index
    df["time"] = pd.to_datetime(df["time"], unit="ns", errors="coerce")
    df = df.set_index("time").sort_index()

    # dropping seconds_elapsed; not needed once we use absolute time
    if "seconds_elapsed" in df.columns:
        df = df.drop(columns=["seconds_elapsed"])

    # renaming axis columns
    df = df.rename(columns={
        "x": f"{prefix}_x",
        "y": f"{prefix}_y",
        "z": f"{prefix}_z",
    })

    return df


def resample_sensor(df):
    """
    Resampling a datetime-indexed sensor dataframe to TARGET_FS (100 Hz)
    and interpolating short gaps.
    """
    return df.resample(f"{TARGET_DT_MS}ms").mean().interpolate(limit=5)


def trim_start_end(df, start_sec=45.0, end_sec=30.0, fs=TARGET_FS):
    """
    Drop the first start_sec seconds and last end_sec seconds based on
    sampling rate fs.
    """
    drop_start = int(start_sec * fs)   # 45 s * 100 Hz = 4500 samples
    drop_end = int(end_sec * fs)       # 30 s * 100 Hz = 3000 samples
    n = len(df)

    if n <= drop_start + drop_end:
        # if session is too short, return empty
        return df.iloc[0:0]

    return df.iloc[drop_start : n - drop_end]


#MAIN LOOP 

if __name__ == "__main__":
    for sess in os.listdir(RAW_ROOT):
        sess_path = os.path.join(RAW_ROOT, sess)
        if not os.path.isdir(sess_path):
            continue

        print("Processing session:", sess)

        acc_path = os.path.join(sess_path, "accelerometer.csv")
        gyr_path = os.path.join(sess_path, "gyroscope.csv")
        grav_path = os.path.join(sess_path, "gravity.csv")
        ori_path  = os.path.join(sess_path, "orientation.csv")

        # skippng if any core sensor file is missing
        if not all(os.path.exists(p) for p in [acc_path, gyr_path, grav_path, ori_path]):
            print("  Missing one or more sensor files, skipping.")
            continue

        # loading each sensor
        acc  = load_sensor_csv(acc_path,  "acc")
        gyr  = load_sensor_csv(gyr_path,  "gyr")
        grav = load_sensor_csv(grav_path, "grav")
        ori  = load_sensor_csv(ori_path,  "ori")

        # resampling to 100 Hz
        acc_r  = resample_sensor(acc)
        gyr_r  = resample_sensor(gyr)
        grav_r = resample_sensor(grav)
        ori_r  = resample_sensor(ori)

        # merging on datetime index
        merged = acc_r.join([gyr_r, grav_r, ori_r], how="outer")
        merged = merged.interpolate(limit=5).dropna()

        # trimming first 45 s and last 30 s
        merged = trim_start_end(merged, start_sec=45.0, end_sec=30.0, fs=TARGET_FS)

        # mild clipping to remove extreme spikes
        if len(merged) > 0:
            merged = merged.clip(
                lower=merged.quantile(0.001),
                upper=merged.quantile(0.999),
                axis=1
            )

        # saving as CSV
        out_path = os.path.join(PROCESSED_SESSION_DIR, f"{sess}.csv")
        merged.to_csv(out_path)
        print("  Saved:", out_path, "rows:", len(merged))
