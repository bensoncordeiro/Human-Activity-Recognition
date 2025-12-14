#deleting unnecessary files like uncalibrated, annotation and totalacceleration files
import os

RAW_ROOT = "zips" 

for sess in os.listdir(RAW_ROOT):
    sess_path = os.path.join(RAW_ROOT, sess)
    if not os.path.isdir(sess_path):
        continue

    for fname in os.listdir(sess_path):
        lower = fname.lower()
        if ("uncalibrated" in lower) or ("annotation" in lower) or ("totalacceleration" in lower):
            full_path = os.path.join(sess_path, fname)
            print("Deleting:", full_path)
            os.remove(full_path)
