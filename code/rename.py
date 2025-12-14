#this code renames all files to standardize it
import os

RAW_ROOT = "zips"  

rename_map = {
    "Accelerometer.csv": "accelerometer.csv",
    "Gyroscope.csv": "gyroscope.csv",
    "Gravity.csv": "gravity.csv",
    "Orientation.csv": "orientation.csv",
    "Metadata.csv": "metadata.csv",
}

for sess in os.listdir(RAW_ROOT):
    sess_path = os.path.join(RAW_ROOT, sess)
    if not os.path.isdir(sess_path):
        continue
    for old, new in rename_map.items():
        src = os.path.join(sess_path, old)
        dst = os.path.join(sess_path, new)
        if os.path.exists(src) and src != dst:
            os.rename(src, dst)
