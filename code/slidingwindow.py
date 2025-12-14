#this script segments processed session CSV files into overlapping windows
import os
import numpy as np
import pandas as pd

PROCESSED_SESSION_DIR = "DATA"
session_index = pd.read_csv("DATA/session_index.csv")

TARGET_FS = 100
WIN_SEC = 2.56
OVERLAP = 0.5

WIN_SIZE = int(WIN_SEC * TARGET_FS)           # 256 samples
STEP = int(WIN_SIZE * (1 - OVERLAP))          # 128 samples

def segment_session_file(csv_path, meta_row, win_size, step):
    df = pd.read_csv(csv_path)

    # if time is present as a column, dropping it so only numeric sensor columns remain
    if "time" in df.columns:
        df = df.drop(columns=["time"])

    data = df.values  # shape [T, C]
    n = data.shape[0]

    X_windows = []
    y_labels = []
    metas = []

    for start in range(0, n - win_size + 1, step):
        end = start + win_size
        seg = data[start:end, :]  # [win_size, C]
        X_windows.append(seg)
        y_labels.append(meta_row["activity"])
        metas.append({
            "person": meta_row["person"],
            "os": meta_row["os"],
            "env": meta_row["env"],
            "placement": meta_row["placement"],
            "session_id": meta_row["session_id"],
            "session_name": meta_row["session_name"],
        })

    return np.array(X_windows), np.array(y_labels), metas

all_X = []
all_y = []
all_meta = []

for _, row in session_index.iterrows():
    csv_path = os.path.join(PROCESSED_SESSION_DIR, row["session_name"])
    if not os.path.exists(csv_path):
        print("Missing:", csv_path)
        continue

    X_sess, y_sess, meta_sess = segment_session_file(
        csv_path, row, WIN_SIZE, STEP
    )
    if X_sess.size == 0:
        continue

    all_X.append(X_sess)
    all_y.append(y_sess)
    all_meta.extend(meta_sess)

X = np.vstack(all_X)          # [N_windows, WIN_SIZE, n_channels]
y = np.concatenate(all_y)     # [N_windows]
meta_df = pd.DataFrame(all_meta)

np.save("DATA/X_windows.npy", X)
np.save("DATA/y_labels.npy", y)
meta_df.to_csv("DATA/window_metadata.csv", index=False)

print("X shape:", X.shape)
print("y shape:", y.shape)
print(meta_df.head())
