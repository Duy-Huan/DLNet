"""Reference baselines on the same split (classical ML), used for the acceptance rule."""

from __future__ import annotations

import time

import numpy as np

from .core import compute_metrics


def run_reference(data: dict) -> list[dict]:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge

    clf = data["task"] == "classification"
    models = (
        {
            "logistic_regression": LogisticRegression(max_iter=3000),
            "random_forest": RandomForestClassifier(300, n_jobs=-1, random_state=0),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=0),
        }
        if clf
        else {
            "ridge": Ridge(),
            "random_forest": RandomForestRegressor(300, n_jobs=-1, random_state=0),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=0),
        }
    )
    out = []
    for name, m in models.items():
        t0 = time.time()
        m.fit(data["Xnp"]["tr"], data["y_raw"]["tr"])
        res = {"name": name, "seconds": round(time.time() - t0, 1)}
        for sp in ("va", "te"):
            pred = m.predict(data["Xnp"][sp])
            res[sp] = compute_metrics(
                data["y_raw"][sp], np.asarray(pred), data["task"], data["primary"]
            )
        out.append(res)
    out.sort(key=lambda r: -r["va"]["score"])
    return out
