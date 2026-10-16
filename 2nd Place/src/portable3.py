"""
portable3.py — version-robust serialization for the v3 submission.

The runtime pins torch 2.12 and leaves sklearn/lightgbm unpinned, so nothing
sklearn-pickled may cross the boundary. Everything here round-trips through
JSON / LightGBM native text / torch state_dicts, and the transforms are
re-implemented in plain numpy so train and inference share one code path.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatPrep:
    """median-impute -> optional log1p -> percentile clip -> standardize.

    Fitted on a training split only; the fitted constants are frozen into the
    submission, so inference stays per-sample independent.
    """

    def __init__(self, log=True, clip_pct=(0.5, 99.5)):
        self.log = log
        self.clip_pct = clip_pct
        self.median = self.lo = self.hi = self.mean = self.std = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        med = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(med), med, 0.0)
        A = self._impute(X)
        if self.log:
            A = np.log1p(np.clip(A, 0, None))
        if self.clip_pct is not None:
            self.lo, self.hi = np.percentile(A, list(self.clip_pct), axis=0)
            A = np.clip(A, self.lo, self.hi)
        self.mean = A.mean(0)
        self.std = A.std(0) + 1e-9
        return self

    def _impute(self, X):
        X = np.array(X, dtype=np.float64, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            idx = np.where(bad)
            X[idx] = np.take(self.median, idx[1])
        return X

    def transform(self, X):
        A = self._impute(X)
        if self.log:
            A = np.log1p(np.clip(A, 0, None))
        if self.lo is not None:
            A = np.clip(A, self.lo, self.hi)
        return (A - self.mean) / self.std

    def to_dict(self):
        return dict(log=bool(self.log),
                    clip_pct=list(self.clip_pct) if self.clip_pct else None,
                    median=self.median.tolist(),
                    lo=None if self.lo is None else self.lo.tolist(),
                    hi=None if self.hi is None else self.hi.tolist(),
                    mean=self.mean.tolist(), std=self.std.tolist())

    @staticmethod
    def from_dict(d):
        p = FeatPrep(log=d["log"], clip_pct=tuple(d["clip_pct"]) if d["clip_pct"] else None)
        p.median = np.array(d["median"])
        p.lo = None if d["lo"] is None else np.array(d["lo"])
        p.hi = None if d["hi"] is None else np.array(d["hi"])
        p.mean = np.array(d["mean"])
        p.std = np.array(d["std"])
        return p


def dump_linear(path: Path, prep: FeatPrep, coef, intercept, features):
    Path(path).write_text(json.dumps(dict(
        prep=prep.to_dict(), coef=np.asarray(coef).ravel().tolist(),
        intercept=float(intercept), features=list(features))))


def load_linear(path: Path):
    d = json.loads(Path(path).read_text())
    d["prep"] = FeatPrep.from_dict(d["prep"])
    d["coef"] = np.array(d["coef"])
    return d


def predict_linear(d, X):
    z = d["prep"].transform(X) @ d["coef"] + d["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def dump_mlp(path: Path, prep: FeatPrep, clf, features):
    """sklearn MLPClassifier -> plain weight matrices (relu hidden, logistic out)."""
    Path(path).write_text(json.dumps(dict(
        prep=prep.to_dict(), features=list(features),
        W=[w.tolist() for w in clf.coefs_],
        b=[v.tolist() for v in clf.intercepts_])))


def load_mlp(path: Path):
    d = json.loads(Path(path).read_text())
    d["prep"] = FeatPrep.from_dict(d["prep"])
    d["W"] = [np.array(w) for w in d["W"]]
    d["b"] = [np.array(v) for v in d["b"]]
    return d


def predict_mlp(d, X):
    h = d["prep"].transform(X)
    n = len(d["W"])
    for i, (W, b) in enumerate(zip(d["W"], d["b"])):
        h = h @ W + b
        if i < n - 1:
            h = np.maximum(h, 0.0)                 # relu on hidden layers only
    return 1.0 / (1.0 + np.exp(-np.clip(h.ravel(), -30, 30)))


def dump_lgbm(path: Path, booster, median, features):
    booster.save_model(str(path))
    Path(str(path) + ".meta.json").write_text(json.dumps(
        dict(median=np.asarray(median).tolist(), features=list(features))))


def load_lgbm(path: Path):
    import lightgbm as lgb
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    return lgb.Booster(model_file=str(path)), np.array(meta["median"]), meta["features"]


def predict_lgbm(booster, median, X):
    X = np.array(X, dtype=np.float64, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        idx = np.where(bad)
        X[idx] = np.take(median, idx[1])
    return booster.predict(X)


def stack(logits: np.ndarray, coef, intercept) -> np.ndarray:
    z = logits @ np.asarray(coef) + float(intercept)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def to_logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))
