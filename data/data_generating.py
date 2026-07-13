"""Generate three datasets: a reference, a no-drift copy, and a data-drift version."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=1234)  # for reproducibility


def data_generation(age_mean, bmi_mean, smoker_prob=0.2, n=2000):
    age = rng.normal(loc=age_mean, scale=9, size=n)
    bmi = rng.normal(loc=bmi_mean, scale=4, size=n)
    # categorical feature
    smoker = rng.choice(["yes", "no"], size=n, p=[smoker_prob, 1 - smoker_prob])

    # NONLINEAR: age enters as a sine wave, not a straight line
    charges = (
        15000
        + 6000 * np.sin(age / 5)
        + 300 * bmi
        + 8000 * (smoker == "yes")
        + rng.normal(0, 1500, n)
    )

    return pd.DataFrame(
        {
            "age": age.round(1),  # continuous feature
            "bmi": bmi.round(1),  # continuous feature
            "smoker": smoker,  # categorical feature
            # the outcome variable
            "charges": charges.round(2),
        }
    )


# baseline
reference = data_generation(age_mean=40, bmi_mean=27, smoker_prob=0.2, n=2000)
reference.to_csv("./data/reference.csv", index=False)

# same distribution --> no drift
no_drift = data_generation(age_mean=40, bmi_mean=27, smoker_prob=0.2, n=1000)
no_drift.to_csv("./data/no_drift.csv", index=False)

# shifted means --> data drift
data_drift = data_generation(age_mean=52, bmi_mean=32, smoker_prob=0.4, n=1000)
data_drift.to_csv("./data/data_drift.csv", index=False)


print("wrote reference.csv, no_drift.csv, data_drift.csv")
