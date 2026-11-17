# GTA VI launch-day Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sandytaztaz-hue/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/sandytaztaz-hue/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for make.fun bounty [`gta6-twitch-ccu-forecast-01`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each hour
of its release day, 2026-11-19 (UTC), after the first — five quantiles
(5%, 25%, 50%, 75%, 95%) per hour, scored by mean pinball loss on the log scale.

## Program

`main.py` (Python 3.13, standard library only). Run:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads `/input/<id>.in` (or `$MAKEFUN_INPUT_DIR`) and writes
`/output/<id>.out`: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`,
23 rows, hours 2–24. Deterministic (two runs byte-identical), ~0.2 s per case.

## Method

Fit at run time from the case's own `comparables` — past big launches' hourly
release-day curves (38 in a practice case, 42 in the final):

- The release-day shape lives in log-ratio space, `ln(v_h / v_1)`. Hour 1's
  print anchors the level; the shape follows the rolling-midnight unlock
  (Asia-Pacific and Europe live by 00:00Z, US East 05:00Z, US West 08:00Z, US
  prime time 18:00–24:00Z): an early dip to ~0.7x at hours 3–10, a rise to
  ~1.5–1.7x at hours 19–23.
- How steep the rise is depends on hour 1's size — a small hour 1 means most
  of the audience has not unlocked yet and the day explodes; a big hour 1
  means most of the world is already watching and the day rises gently. So
  `log(v_1)` is the one feature that matters (hour-1 channels, pre-release
  level and pre-release jump add nothing in leave-one-out tests).
- For each hour, the median log-viewers is a 50/50 blend of (a) a global
  per-hour log regression `ln(v_h) = a_h + b_h * ln(v_1)` (slope clamped to
  [0, 1.5]) and (b) the median log-ratio of the 12 comparables nearest in
  `|ln(v_1) - ln(v_1 target)|`. The blend hedges regression extrapolation
  when the target's hour 1 sits above the training range against the kNN
  saturating at the biggest neighbours' shape.
- Quantiles are Gaussian in log space around that median, with spread = the
  residual sd of the blended model on the kNN neighbourhood (each neighbour's
  log-viewers minus the blend's prediction at that neighbour's hour-1 level),
  shrunk 50/50 toward a smooth prior `0.22 + 0.02*sqrt(h-1)` that widens with
  horizon. Calibrating the spread to the shipped blend (not to the kNN half
  alone) keeps the 5%/95% quantiles honest.

## Validation

- Leave-one-out on the 27 big comparables (hour-1 viewers ≥ 20k, the GTA
  regime): mean pinball loss ~0.25, vs 0.93 for the house persistence baseline
  and 0.71 for a plain shape-median. On the biggest launches the loss is
  0.11–0.16.
- Local checks on the kit's four practice cases: valid format on all four,
  byte-identical repeat runs, 0.2 s per case (limit 8 s).
- Dev score on the make.fun leaderboard (practice cases) is the official
  check; see `BUILD_LOG.md` in the work directory for the submitted result.
