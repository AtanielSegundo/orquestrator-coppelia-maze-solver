"""
One CSV row per cell arrival: what each corrector measured and did, and the best
estimate of how far from the cell centre the kart really was.

The belief right after a leg is the goto's target, so hypot(belief - centre) is
~0 by construction and measures nothing. The arrival error logged here is the
camera-CORRECTED estimate minus the centre (longitudinal from Recenter's first
measurement when the camera could not see that axis) -- the number that decides
whether the corrections are working, and the one the TCC plots.
"""
from __future__ import annotations

import csv
import math
import os
import time
from typing import Dict, List, Optional

FIELDS = [
    "t", "kind", "cell", "heading_deg",
    "cam_outcome", "cam_reason", "cam_views", "cam_hits", "cam_rms_px",
    "cam_long_cm", "cam_lat_cm", "cam_yaw_deg",
    "cam_std_long_cm", "cam_std_lat_cm", "cam_std_yaw_deg",
    "err_long_cm", "err_lat_cm", "err_xy_cm", "err_long_source",
    "moved", "move_mode", "move_blocked",
    "verify_outcome", "verify_lat_cm", "verify_yaw_deg",
    "recenter_first_cm", "recenter_last_cm", "recenter_moves",
    "sweep_used", "sweep_lateral_cm", "duration_s",
]


def _cm(v) -> str:
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v * 100:.2f}"


def _deg(v) -> str:
    return "" if v is None or (isinstance(v, float) and not math.isfinite(v)) \
        else f"{math.degrees(v):.2f}"


class ArrivalLog:
    def __init__(self, directory: str, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled
        self.path: Optional[str] = None
        self.rows: List[Dict] = []

    def record(self, row: Dict) -> None:
        self.rows.append(row)
        if not self.enabled:
            return
        if self.path is None:
            os.makedirs(self.directory, exist_ok=True)
            self.path = os.path.join(self.directory,
                                     time.strftime("arrivals_%Y%m%d_%H%M%S.csv"))
            with open(self.path, "w", newline="", encoding="utf-8") as fp:
                csv.writer(fp).writerow(FIELDS)
        with open(self.path, "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([row.get(k, "") for k in FIELDS])

    def summary(self) -> str:
        errs = [float(r["err_xy_cm"]) for r in self.rows
                if r.get("err_xy_cm") not in ("", None)]
        lats = [abs(float(r["err_lat_cm"])) for r in self.rows
                if r.get("err_lat_cm") not in ("", None)]
        cam = [r for r in self.rows if r.get("cam_outcome")]
        refused = sum(1 for r in cam if r["cam_outcome"] == "recusado")
        sweeps = sum(1 for r in self.rows if r.get("sweep_used") == "1")
        if not self.rows:
            return "nenhuma chegada registrada"
        out = (f"{len(self.rows)} chegada(s), camera em {len(cam)} "
               f"({refused} recusa(s)), varredura em {sweeps}")
        if errs:
            errs.sort()
            p90 = errs[min(len(errs) - 1, int(round(0.9 * (len(errs) - 1))))]
            out += (f" | erro ate o centro: media {sum(errs) / len(errs):.1f} cm, "
                    f"p90 {p90:.1f} cm, max {max(errs):.1f} cm")
        if lats:
            out += f", |lateral| max {max(lats):.1f} cm"
        if self.path:
            out += f" ({self.path})"
        return out
