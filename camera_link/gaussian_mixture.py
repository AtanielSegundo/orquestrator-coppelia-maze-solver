"""
A small Gaussian mixture, fitted by EM, in plain numpy.

The OpenCV build in use (5.0) ships without `cv2.ml`, so this is the colour
model's only dependency. It is only ever fitted on a few thousand 3-D colour
samples, so clarity beats speed here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class GaussianMixture:
    def __init__(self, weights, means, covs):
        self.weights = np.asarray(weights, dtype=float)
        self.means = np.asarray(means, dtype=float)
        self.covs = np.asarray(covs, dtype=float)

    @classmethod
    def fit(cls, X: np.ndarray, k: int = 2, iters: int = 40,
            reg: float = 1.0, seed: int = 0) -> Optional["GaussianMixture"]:
        """EM from a farthest-point initialisation. `reg` is added to every
        covariance diagonal (in the feature's own units), so a component that
        collapses onto a flat patch of colour stays a usable Gaussian."""
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if n < max(10, 3 * k):
            return None
        k = max(1, min(k, n // 10))
        rng = np.random.default_rng(seed)
        centres = [X[rng.integers(n)]]
        for _ in range(1, k):
            dist = np.min([np.sum((X - c) ** 2, axis=1) for c in centres], axis=0)
            centres.append(X[int(np.argmax(dist))])
        means = np.array(centres)
        covs = np.array([np.cov(X.T) + reg * np.eye(d) for _ in range(k)])
        weights = np.full(k, 1.0 / k)
        model = cls(weights, means, covs)
        prev = -np.inf
        for _ in range(iters):
            logp = model._component_logpdf(X) + np.log(model.weights)[None, :]
            top = logp.max(axis=1, keepdims=True)
            ll = top[:, 0] + np.log(np.exp(logp - top).sum(axis=1))
            resp = np.exp(logp - ll[:, None])
            nk = resp.sum(axis=0) + 1e-9
            model.weights = nk / n
            model.means = (resp.T @ X) / nk[:, None]
            for j in range(k):
                diff = X - model.means[j]
                model.covs[j] = (resp[:, j, None] * diff).T @ diff / nk[j] + reg * np.eye(d)
            total = float(ll.mean())
            if total - prev < 1e-4:
                break
            prev = total
        return model

    def _component_logpdf(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        out = np.empty((n, len(self.weights)))
        for j, (mu, cov) in enumerate(zip(self.means, self.covs)):
            inv = np.linalg.inv(cov)
            _sign, logdet = np.linalg.slogdet(cov)
            diff = X - mu
            maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
            out[:, j] = -0.5 * (maha + logdet + d * np.log(2 * np.pi))
        return out

    def loglik(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logp = self._component_logpdf(X) + np.log(self.weights)[None, :]
        top = logp.max(axis=1, keepdims=True)
        return top[:, 0] + np.log(np.exp(logp - top).sum(axis=1))

    def to_dict(self) -> dict:
        return {"w": self.weights.tolist(), "mu": self.means.tolist(),
                "cov": self.covs.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianMixture":
        return cls(d["w"], d["mu"], d["cov"])
