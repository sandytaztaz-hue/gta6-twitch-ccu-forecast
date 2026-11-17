"""GTA VI launch-day Twitch CCU forecaster (make.fun gta6-twitch-ccu-forecast-01).

Reads one case (/input/<id>.in) and writes /output/<id>.out: for each of hours
2..24 of the release day (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of that
hour's concurrent Twitch viewers in the game's category.

Method — fit at run time from the case's own `comparables` (past big launches'
hourly release-day curves, 38 in a practice case, 42 in the final):

  The release-day shape lives in log-ratio space, r_h = ln(v_h / v_1). Hour 1's
  print anchors the level; the shape is driven by the rolling-midnight unlock
  (Asia-Pacific and Europe live by 00:00Z, US East 05:00Z, US West 08:00Z, US
  prime time 18:00-24:00Z): an early dip to ~0.7x at hours 3-10, a rise to
  ~1.5-1.7x at hours 19-23. How steep that rise is depends on hour 1's size —
  a small hour 1 means most of the audience has not unlocked yet (or a
  simultaneous global unlock later in the day) and the day explodes; a big
  hour 1 means most of the world is already watching and the day rises gently.
  So log(v_1) is the one feature that matters (channels, pre-release level and
  pre-release jump add nothing in leave-one-out tests on the comparables).

  For each hour h = 2..24 the median log-viewers is a 50/50 blend of two
  estimators, both fit on the comparables that have hour h:
    (a) global per-hour log regression: ln(v_h) = a_h + b_h * ln(v_1),
        slope clamped to [0, 1.5];
    (b) kNN median: the 12 comparables nearest in |ln(v_1) - ln(v_1 target)|,
        median of their log-ratios ln(v_h / v_1), added to ln(v_1 target).
  The blend hedges regression extrapolation when the target's hour 1 sits
  above the training range (largest comparable hour 1 is ~875k) against the
  kNN saturating at the biggest neighbours' shape.

  Quantiles are Gaussian in log space around that median. The spread is the
  kNN neighbourhood's residual standard deviation, shrunk 50/50 toward a
  smooth prior that widens with horizon, sd_prior(h) = 0.22 + 0.02*sqrt(h-1):
  twelve neighbours give noisy empirical tails, so the prior keeps the
  5%/95% quantiles honest instead of min/max-driven.

  Fallbacks: with fewer than 8 comparables for an hour, use the global median
  shape of all comparables; with no comparables at all, persistence (the house
  baseline shape, sigma 0.15). All quantiles are rounded half-up to 2 decimals,
  clamped to (0, 1e9] and forced non-decreasing.

Standard library only. Deterministic: candidates are ordered by
(distance, name), so two runs write identical bytes. Runs in well under a
second per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Blend weight of the kNN median against the global regression (0.5 / 0.5).
KNN_WEIGHT = 0.5
# Neighbours for the local median and residual spread.
K_NEIGHBOURS = 12
# Minimum comparables with hour h before the regression/kNN are trusted.
MIN_COMPS = 8
# Smooth spread prior, in log units, widening with horizon.
SIGMA_PRIOR_BASE = 0.22
SIGMA_PRIOR_SLOPE = 0.02
# Floor on the final spread so quantiles never collapse onto the median.
SIGMA_FLOOR = 0.08
# Persistence fallback spread (the house baseline's sigma).
PERSIST_SIGMA = 0.15


def parse_ts(text):
    """ISO 8601 timestamp to an aware datetime (UTC)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def ols_slope_intercept(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b) with b clamped to [0, 1.5]."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 1.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    slope = max(0.0, min(1.5, slope))
    return my - slope * mx, slope


def median(values):
    """Median of a non-empty sorted list."""
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    return 0.5 * (values[n // 2 - 1] + values[n // 2])


def extract_curves(case):
    """Comparable launch curves: list of {name, log_v1, hours: {h: log viewers}}.

    A comparable's hourly points span 14 days before to 3 days after its
    release day; hour h of the release day is the point in [day_start + (h-1)h,
    day_start + h*h). Points with no viewers are skipped.
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours = {}
        for point in comp.get("hourly") or []:
            viewers = point.get("viewers")
            if not viewers or viewers <= 0:
                continue
            try:
                at = parse_ts(point["from"])
            except (KeyError, TypeError, ValueError):
                continue
            hour = int((at - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                hours[hour] = math.log(viewers)
        if 1 in hours:
            curves.append(
                {"name": comp.get("name", ""), "log_v1": hours[1], "hours": hours}
            )
    return curves


def hour_forecast(log_v1, hour, curves):
    """Median log-viewers and spread for one hour, from the comparables."""
    have = [c for c in curves if hour in c["hours"]]
    if not have:
        # No comparable has this hour: persistence around hour 1.
        return log_v1, PERSIST_SIGMA * math.sqrt(hour - 1)
    if len(have) < MIN_COMPS:
        # Too few: the global median shape of every comparable with this hour.
        ratios = sorted(c["hours"][hour] - c["log_v1"] for c in have)
        return log_v1 + median(ratios), 0.30 + 0.02 * math.sqrt(hour - 1)

    # (a) Global per-hour log regression on ln(v_1).
    xs = [c["log_v1"] for c in have]
    ys = [c["hours"][hour] for c in have]
    intercept, slope = ols_slope_intercept(xs, ys)
    reg_pred = intercept + slope * log_v1

    # (b) kNN median: nearest neighbours in |ln(v_1) - ln(v_1 target)|,
    # ordered by (distance, name) so the choice is deterministic.
    ranked = sorted(have, key=lambda c: (abs(c["log_v1"] - log_v1), c["name"]))
    neighbours = ranked[:K_NEIGHBOURS]
    ratios = sorted(c["hours"][hour] - c["log_v1"] for c in neighbours)
    knn_med = median(ratios)
    knn_pred = log_v1 + knn_med

    med = KNN_WEIGHT * knn_pred + (1.0 - KNN_WEIGHT) * reg_pred
    # Residual spread around the BLENDED model, evaluated at each neighbour:
    # each neighbour's own log-viewers minus the blend's prediction at that
    # neighbour's hour-1 level. This calibrates the quantiles to the model
    # actually shipped, not to the kNN half alone.
    resid = []
    for c in neighbours:
        blend_at_c = KNN_WEIGHT * (c["log_v1"] + knn_med) + (1.0 - KNN_WEIGHT) * (
            intercept + slope * c["log_v1"]
        )
        resid.append(c["hours"][hour] - blend_at_c)
    resid_sd = math.sqrt(sum(r * r for r in resid) / max(1, len(resid) - 1))
    prior = SIGMA_PRIOR_BASE + SIGMA_PRIOR_SLOPE * math.sqrt(hour - 1)
    sd = max(SIGMA_FLOOR, 0.5 * resid_sd + 0.5 * prior)
    return med, sd


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    log_v1 = math.log(max(1, first["viewers"]))
    curves = extract_curves(case)

    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        med, sd = hour_forecast(log_v1, hour, curves)
        quantiles = []
        prev = 0.0
        for z in Z:
            q = round2(math.exp(med + z * sd))
            q = max(q, 0.01)
            q = min(q, 1e9)
            # Quantiles must be non-decreasing; rounding can tie them, never invert.
            if q < prev:
                q = prev
            quantiles.append(q)
            prev = q
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
