"""Replay a data batch through the /predict API so the dashboard reflects it.

Compare how the predicted-charge panel shifts between batches:
    python replay_batch.py --csv data/reference.csv    # normal
    python replay_batch.py --csv data/data_drift.csv    # drifted -> distribution shifts
    python replay_batch.py --csv data/data_drift.csv --delay 0.2
"""

import argparse
import time

import pandas as pd
import requests

URL = "http://localhost:8000/predict"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--delay", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=1000)
    args = p.parse_args()

    df = pd.read_csv(args.csv).head(args.limit)
    ok, charges = 0, []
    for i, row in df.iterrows():
        payload = {"age": int(row["age"]), "bmi": float(row["bmi"]), "smoker": str(row["smoker"])}
        try:
            r = requests.post(URL, json=payload, timeout=5)
            if r.ok:
                ok += 1
                charges.append(r.json()["charge"])
        except Exception as e:
            print("err:", e)
        if i % 100 == 0:
            avg = sum(charges) / len(charges) if charges else 0
            print(f"{i:4d}  sent={ok}  avg_charge={avg:.0f}")
        time.sleep(args.delay)

    avg = sum(charges) / len() if charges else 0
    print(f"\nDone: {ok} predictions from {args.csv}, avg charge = {avg:.0f}")


if __name__ == "__main__":
    main()
