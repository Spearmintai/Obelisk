"""Comparing two device profiles that were never in the same room.

A profile is a complete record of one device's behaviour.  Once two profiles
exist, the question "will this run the same way over there" is answered by
reading two files, with no hardware present and nothing to schedule.  That is the
property that makes the measurements worth keeping: the hardware time is spent
once per device, and every pairing after that is free.
"""

from .compare import compare_probes, check_internal, grade
from .fingerprint import compare_fingerprints


STATUS_RANK = {"EXACT": 0, "CLOSE": 1, "MISMATCH": 2, "SHAPE": 2,
               "MISSING": 3, "ERROR": 3}

# PyTorch exposes the same operation under several names, and again as an
# in-place variant.  Counting those separately inflates a finding; counting the
# distinct operations is the number that means something.
ALIASES = {
    "arcsin": "asin", "arccos": "acos", "arctan": "atan", "arcsinh": "asinh",
    "arccosh": "acosh", "arctanh": "atanh", "absolute": "abs", "divide": "div",
    "multiply": "mul", "subtract": "sub", "true_divide": "div",
    "negative": "neg", "greater": "gt", "less": "lt", "not_equal": "ne",
    "fix": "trunc", "clip": "clamp", "ger": "outer", "row_stack": "vstack",
}


def canonical(op):
    _, _, name = op.partition(".")
    name = name.rstrip("_")
    return ALIASES.get(name, name)


def distinct(ops):
    return sorted({canonical(o) for o in ops})


def diff_coverage(ref_cov, got_cov):
    """Which operators changed status between two devices."""
    ref_ops = {r["op"]: r for r in (ref_cov or {}).get("operators", [])}
    got_ops = {r["op"]: r for r in (got_cov or {}).get("operators", [])}

    lost, degraded, gained, same = [], [], [], 0
    for op, r in sorted(ref_ops.items()):
        g = got_ops.get(op)
        if g is None:
            continue
        rs, gs = r.get("status"), g.get("status")
        if rs == gs and rs in ("EXACT", "CLOSE"):
            same += 1
            continue
        rr, gr = STATUS_RANK.get(rs, 9), STATUS_RANK.get(gs, 9)
        row = {"op": op, "reference": rs, "target": gs,
               "detail": g.get("detail"), "error": g.get("error")}
        if gr >= 3 > rr:
            lost.append(row)
        elif gr > rr:
            degraded.append(row)
        elif gr < rr:
            gained.append(row)
        else:
            same += 1

    return {
        "unavailable_on_target": lost,
        "numerically_different": degraded,
        "better_on_target": gained,
        "unchanged": same,
        "compared": len(set(ref_ops) & set(got_ops)),
        "distinct_unavailable": distinct([r["op"] for r in lost]),
        "distinct_wrong": distinct([r["op"] for r in degraded
                                    if r.get("target") in ("MISMATCH", "SHAPE")]),
    }


def diff_dtypes(ref_dt, got_dt):
    rows = []
    for name, r in sorted((ref_dt or {}).items()):
        g = (got_dt or {}).get(name)
        if g is None:
            continue
        missing = [k for k, v in r.items()
                   if v is True and g.get(k) is not True and k != "detail"]
        if missing:
            rows.append({"dtype": name, "unsupported": missing})
    return rows


def compare_profiles(ref, got, tol=None):
    """Compare two recorded profiles without touching either device."""
    rp, gp = ref.get("payload", {}), got.get("payload", {})

    probe_results = compare_probes(rp.get("probes", {}), gp.get("probes", {}), tol)
    internal = check_internal(gp.get("probes", {}))
    coverage = diff_coverage(rp.get("coverage"), gp.get("coverage"))
    dtypes = diff_dtypes(rp.get("dtypes"), gp.get("dtypes"))

    g = grade(probe_results, internal)
    if coverage["unavailable_on_target"]:
        n = len(coverage["unavailable_on_target"])
        g["reasons"].insert(0, {
            "grade": "D",
            "reason": "%d operator%s available on the reference %s not run here"
                      % (n, "" if n == 1 else "s", "does" if n == 1 else "do"),
        })
        if g["grade"] in ("A", "B", "C"):
            g["grade"] = "D"

    return {
        "reference": _describe(ref),
        "target": _describe(got),
        "reference_digest": ref.get("digest"),
        "target_digest": got.get("digest"),
        "probes": probe_results,
        "internal": internal,
        "coverage": coverage,
        "dtypes": dtypes,
        "grade": g,
    }


def _describe(bp):
    d = bp.get("device", {})
    return "%s (%s, torch %s)" % (d.get("name", "?"), d.get("backend", "?"),
                                  d.get("torch", "?"))
