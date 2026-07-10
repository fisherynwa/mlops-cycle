"""Fire randomized predictions at the API so Prometheus/Grafana have data to plot.

Run (with the stack up):
    python simulate_traffic.py                 # 500 requests, default
    python simulate_traffic.py --n 2000 --delay 0.05
"""

import argparse
import random
import time

import requests

URL = "http://localhost:8000/predict"


def one_request():
    payload = {
        "age": random.randint(18, 70),
        "bmi": round(random.uniform(18, 45), 1),
        "smoker": random.choice(["yes", "no"]),
    }
    try:
        r = requests.post(URL, json=payload, timeout=5)
        return r.status_code, r.json().get("charge") if r.ok else None
    except Exception as e:
        return None, str(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500, help="number of requests")
    p.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    args = p.parse_args()

    ok = 0
    for i in range(args.n):
        code, charge = one_request()
        if code == 200:
            ok += 1
        if i % 50 == 0:
            print(f"{i:4d}/{args.n}  last: {code} charge={charge}")
        time.sleep(args.delay)
    print(f"\nDone: {ok}/{args.n} succeeded")


if __name__ == "__main__":
    main()
