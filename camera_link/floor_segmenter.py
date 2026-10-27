"""
Wall vs floor, per pixel, with a colour model learned from the frame itself.

Self-supervised, per view
-------------------------
There is no fixed colour threshold. Every time a view is captured, the caller
also has a PREDICTION of what that view should show -- the map rendered from the
believed pose (runner/reference_renderer.py). The pose is off by a few
centimetres at most, so the prediction is wrong only in a thin band around each
predicted wall/floor boundary. Eroding both predicted regions by
`margin_px` leaves pixels whose label is right even when the pose is not, and
those pixels train a small Gaussian mixture per class on the frame being
classified.

That buys three things a calibrated threshold cannot:
* no colour calibration step at all -- the boxes and the floor are learned
  wherever the kart is;
* auto-exposure and white balance changes between views are absorbed, since
  each view trains on itself;
* a scene where walls and floor are NOT separable in colour is detected
  (training accuracy below `min_accuracy`) and refused, instead of producing a
  confident wrong profile.

Features are CIE Lab with lightness down-weighted (`l_weight`): shadows and
exposure mostly move L*, while the cardboard-vs-floor difference lives in a*/b*.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

from camera_link.gaussian_mixture import GaussianMixture

FLOOR_LABEL = 0
WALL_LABEL = 1


class SegmenterModel(NamedTuple):
    wall: GaussianMixture
    floor: GaussianMixture
    log_prior_ratio: float
    accuracy: float              # balanced, on the training pixels


class FloorSegmenter:
    def __init__(self, components: int = 2, l_weight: float = 1.0 / 3.0,
                 margin_px: int = 10, min_pixels: int = 300,
                 max_pixels: int = 4000, min_accuracy: float = 0.85,
                 refine_iters: int = 2, refine_llr: float = 2.0,
                 llr_clip: float = 4.0, seed: int = 0):
        self.components = components
        self.l_weight = l_weight
        self.margin_px = margin_px
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.min_accuracy = min_accuracy
        self.refine_iters = refine_iters
        self.refine_llr = refine_llr
        self.llr_clip = llr_clip
        self._rng = np.random.default_rng(seed)

    def features(self, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        lab[..., 0] *= self.l_weight
        return lab

    def train(self, bgr: np.ndarray, labels: np.ndarray,
              exclude: Optional[np.ndarray] = None
              ) -> Tuple[Optional[SegmenterModel], str]:
        """Fit the two colour models, seeded by predicted `labels` (H, W).

        Seed: FLOOR_LABEL and WALL_LABEL pixels, each eroded by `margin_px` so
        the band a wrong pose mislabels never reaches the training set.

        Then self-training: the pixels the seeded model itself classifies with
        confidence (|LLR| > `refine_llr`) become the next training set, for
        `refine_iters` rounds. Within the region the map calls maze (wall or
        floor, un-eroded) -- the prediction decides WHERE to look, the colours
        decide WHAT each pixel is. That is what keeps the boundary unbiased
        when the believed pose is off by more than the margin covers: measured
        in the closed loop, a 2.7 deg yaw error shifts vertical wall edges by
        ~19 px, the seed carries those mislabelled pixels, and the profile came
        out 2-3 px off with the seed alone and ~0.3 px after refining.

        The accuracy gate is checked against the SEED, the only labels that do
        not come from the model being judged. Returns (model, "") or
        (None, reason).
        """
        feats = self.features(bgr)
        flat = feats.reshape(-1, 3)
        k = 2 * self.margin_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        samples = []
        for label in (WALL_LABEL, FLOOR_LABEL):
            mask = (labels == label).astype(np.uint8)
            mask = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT,
                             borderValue=0).astype(bool)
            if exclude is not None:
                mask &= ~exclude
            idx = np.flatnonzero(mask)
            if idx.size < self.min_pixels:
                name = "parede" if label == WALL_LABEL else "chao"
                return None, f"poucos pixels de {name} para treinar ({idx.size})"
            if idx.size > self.max_pixels:
                idx = self._rng.choice(idx, self.max_pixels, replace=False)
            samples.append(flat[idx])
        wall_x, floor_x = samples
        wall = GaussianMixture.fit(wall_x, self.components)
        floor = GaussianMixture.fit(floor_x, self.components)
        if wall is None or floor is None:
            return None, "modelo de cor nao convergiu"
        prior = 0.0
        model = SegmenterModel(wall, floor, prior, 0.0)

        roi = (labels == WALL_LABEL) | (labels == FLOOR_LABEL)
        if exclude is not None:
            roi &= ~exclude
        roi_idx = np.flatnonzero(roi)
        for _ in range(self.refine_iters):
            llr = self._llr_flat(flat[roi_idx], model)
            refit = []
            for idx in (roi_idx[llr > self.refine_llr], roi_idx[llr < -self.refine_llr]):
                if idx.size < self.min_pixels:
                    break
                if idx.size > self.max_pixels:
                    idx = self._rng.choice(idx, self.max_pixels, replace=False)
                refit.append(GaussianMixture.fit(flat[idx], self.components))
            if len(refit) < 2 or refit[0] is None or refit[1] is None:
                break
            model = SegmenterModel(refit[0], refit[1], prior, 0.0)

        acc_wall = float(np.mean(self._llr_flat(wall_x, model) > 0))
        acc_floor = float(np.mean(self._llr_flat(floor_x, model) < 0))
        accuracy = 0.5 * (acc_wall + acc_floor)
        model = model._replace(accuracy=accuracy)
        if accuracy < self.min_accuracy:
            return None, (f"parede e chao com cores parecidas demais "
                          f"(acerto {accuracy * 100:.0f}%)")
        return model, ""

    def _llr_flat(self, X: np.ndarray, model: SegmenterModel) -> np.ndarray:
        return model.wall.loglik(X) - model.floor.loglik(X) + model.log_prior_ratio

    def llr(self, bgr: np.ndarray, model: SegmenterModel) -> np.ndarray:
        """Log-likelihood ratio wall vs floor, float32 (H, W), lightly blurred
        so single-pixel noise does not flip a label.

        Clipped SYMMETRICALLY to +-`llr_clip` before the blur. Unclipped (or
        clipped wide), a confident class (+50) next to a less confident one
        (-10) averages to a zero crossing pushed into the weaker side, so the
        boundary moves by a pixel or two depending on how sure each class
        happens to be -- measured 1.4-2.2 px per view in the closed loop,
        about 1 px with the clip at 4."""
        feats = self.features(bgr)
        h, w, _ = feats.shape
        out = self._llr_flat(feats.reshape(-1, 3), model).reshape(h, w)
        out = np.clip(out, -self.llr_clip, self.llr_clip).astype(np.float32)
        return cv2.GaussianBlur(out, (5, 5), 0)

    @staticmethod
    def wall_mask(llr: np.ndarray) -> np.ndarray:
        mask = (llr > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask.astype(bool)
