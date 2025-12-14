#this script builds an index of all processed session CSV files
import os
import re
import pandas as pd

PROCESSED_SESSION_DIR = "DATA"

pattern = re.compile(
    r"^(?P<person>[^_]+)_(?P<os>[^_]+)_(?P<activity>[^_]+)_(?P<env>[^_]+)_(?P<placement>[^_]+)_(?P<session_id>[^-]+)-(?P<date>[\d-]+)_(?P<time>[\d-]+)\.csv$"
)

rows = []
for fname in os.listdir(PROCESSED_SESSION_DIR):
    m = pattern.match(fname)
    if not m:
        print("Skipping:", fname)
        continue
    info = m.groupdict()
    info["session_name"] = fname
    info["start_datetime"] = pd.to_datetime(
        info["date"] + " " + info["time"].replace("-", ":"), errors="coerce"
    )
    rows.append(info)

session_index = pd.DataFrame(rows)
session_index.to_csv("DATA/session_index.csv", index=False)
session_index.head()
