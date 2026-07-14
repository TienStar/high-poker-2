"""Feature views for high-poker-2 (published; train == serve exactly).

tree_view: per-chunk behavioral aggregates + bucket/entropy fingerprints from
           the base library, plus chunk-size descriptors (raw and log) so the
           model learns the group-size axis (live groups are larger than
           benchmark groups), plus this miner's TAIL lens.
wide_view: tree_view merged with the v2 order-statistic aggregates into one
           deduplicated dictionary (the neural member consumes this union).

Tail lens (high-poker-2 only): the base library reports each behavioral metric
as a fixed stat panel (min/q10/q50/q90/max/mean/std). The bulk of that panel
describes the middle of the distribution, but automation tends to betray itself
at the edges - a bot's rare hands look like its common ones. For every metric
this view adds the edge geometry the panel implies but does not state:

  __range   = max - min                          raw support width;
  __tail_hi = (max - q90) / |q90 - q50|          upper-tail stretch vs the
                                                 upper-mid step;
  __tail_lo = (q10 - min) / |q50 - q10|          lower-tail stretch;
  __cv      = std / |mean|                       coefficient of variation.

The three ratios are scale-free, so they transfer across the benchmark(30-40)
-> live(80-105) group-size shift; __range is deliberately left in raw units as
the one absolute-scale member of the lens.
"""
import math

from features_v2 import extract_features_v2
from poker44_ml.features import chunk_features

_EPS = 1e-6
_CLIP = 1e6
_STATS = ("min", "q10", "q25", "q50", "q75", "q90", "max", "mean", "std")


def _finite(x):
    """Clip to a finite range: mat() fills NaN but NOT inf, and an inf would
    poison StandardScaler -> PCA -> MLP for the whole batch."""
    if x != x or x == float("inf") or x == float("-inf"):
        return 0.0
    return max(-_CLIP, min(_CLIP, float(x)))


def _stat_panels(d):
    """Group `{metric}_{stat}` keys back into {metric: {stat: value}}."""
    panels = {}
    for key, value in d.items():
        for stat in _STATS:
            suffix = "_" + stat
            if key.endswith(suffix):
                panels.setdefault(key[: -len(suffix)], {})[stat] = value
                break
    return panels


def _tail_terms(d):
    out = {}
    for metric, panel in _stat_panels(d).items():
        lo, hi = panel.get("min"), panel.get("max")
        q10, q50, q90 = panel.get("q10"), panel.get("q50"), panel.get("q90")
        mean, std = panel.get("mean"), panel.get("std")
        if lo is not None and hi is not None:
            out[metric + "__range"] = _finite(hi - lo)
        if hi is not None and q90 is not None and q50 is not None:
            out[metric + "__tail_hi"] = _finite((hi - q90) / (abs(q90 - q50) + _EPS))
        if lo is not None and q10 is not None and q50 is not None:
            out[metric + "__tail_lo"] = _finite((q10 - lo) / (abs(q50 - q10) + _EPS))
        if std is not None and mean is not None:
            out[metric + "__cv"] = _finite(std / (abs(mean) + _EPS))
    return out


def tree_view(chunk):
    hands = chunk or []
    d = chunk_features(hands)
    n = float(len(hands))
    d["hand_count"] = n
    d["hand_count_log"] = math.log1p(n)
    d.update(_tail_terms(d))
    return d


def wide_view(chunk):
    d = dict(extract_features_v2(chunk or []))
    d.update(tree_view(chunk))
    return d
