"""Numerical fingerprints.

A fingerprint is a small, portable, exactly-comparable description of a tensor.
It is small enough to travel in a text file, which is what lets a reference
device hand its results to a target device that shares no filesystem and has no
network.

Three layers, cheapest first:

  digest   blake2b over the raw bytes.  Equal digests mean bit-identical.
  stats    mean/std/min/max/absmax/l2 in float64, plus NaN and infinity counts.
  sample   a deterministic stride-sample of the values, so two fingerprints can
           be compared elementwise without either tensor being present.

The sample is what makes a fingerprint a measurement rather than a checksum: a
checksum tells you something changed, a sample tells you by how much.
"""

import hashlib
import math

SAMPLE_SIZE = 512


def _sample_positions(numel, k=SAMPLE_SIZE):
    if numel <= k:
        return list(range(numel))
    step = numel / float(k)
    return [min(numel - 1, int(i * step)) for i in range(k)]


def fingerprint(tensor, sample_size=SAMPLE_SIZE):
    """Describe a torch tensor in a form that survives a trip through JSON."""
    import torch

    t = tensor.detach()
    out = {
        "dtype": str(t.dtype).replace("torch.", ""),
        "shape": list(t.shape),
        "numel": int(t.numel()),
        "device": str(t.device),
    }
    if t.numel() == 0:
        out["digest"] = "empty"
        out["stats"] = {}
        out["sample"] = []
        return out

    flat = t.to("cpu").contiguous().reshape(-1)

    try:
        raw = flat.view(torch.uint8).numpy().tobytes()
        out["digest"] = hashlib.blake2b(raw, digest_size=16).hexdigest()
        out["nbytes"] = len(raw)
    except Exception:
        out["digest"] = "unhashable"

    f = flat.to(torch.float64)
    finite = torch.isfinite(f)
    n_nan = int(torch.isnan(f).sum())
    n_inf = int((~finite & ~torch.isnan(f)).sum())
    clean = f[finite]
    stats = {"nan": n_nan, "inf": n_inf}
    if clean.numel():
        stats.update({
            "mean": float(clean.mean()),
            "std": float(clean.std(unbiased=False)) if clean.numel() > 1 else 0.0,
            "min": float(clean.min()),
            "max": float(clean.max()),
            "absmax": float(clean.abs().max()),
            "l2": float(clean.double().pow(2).sum().sqrt()),
        })
    out["stats"] = stats

    pos = _sample_positions(int(flat.numel()), sample_size)
    idx = torch.tensor(pos, dtype=torch.long)
    out["sample"] = [float(v) for v in f[idx]]
    out["sample_positions"] = pos
    return out


def _rel(a, b):
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


# Tolerance is expressed against the scale of the tensor, not against each
# individual element.  Two values of 1e-41 that differ by 1e-41 are not a
# hundred percent apart in any sense that matters; they are both zero.
DTYPE_TOLERANCE = {
    "float64": 1e-12,
    "float32": 1e-5,
    "float16": 1e-2,
    "bfloat16": 5e-2,
}


def tolerance_for(dtype, override=None):
    if override is not None:
        return override
    return DTYPE_TOLERANCE.get(dtype, 1e-5)


def compare_fingerprints(ref, got, tol=None, expect="numeric", info=False):
    """Compare two fingerprints.

    The headline number is ``scaled_diff``: the largest absolute disagreement,
    divided by the size of the largest value in the reference.  It answers "how
    big is the error compared with the thing being measured", which is the only
    form of the question that stays meaningful when some of the values are zero.

    ``expect="bitwise"`` is for quantities that have no excuse to differ at all,
    such as a random number stream or the order of a sort.
    """
    result = {
        "dtype": ref.get("dtype"),
        "shape_match": ref.get("shape") == got.get("shape"),
        "dtype_match": ref.get("dtype") == got.get("dtype"),
        "bit_identical": bool(ref.get("digest") and ref.get("digest") == got.get("digest")),
        "expect": expect,
    }
    tol = tolerance_for(ref.get("dtype", ""), tol)
    result["tolerance"] = tol

    rs, gs = ref.get("stats") or {}, got.get("stats") or {}

    if info:
        result["verdict"] = "INFO"
        result["reference_value"] = rs.get("absmax", rs.get("mean"))
        result["target_value"] = gs.get("absmax", gs.get("mean"))
        return result

    if not result["shape_match"]:
        result["verdict"] = "SHAPE"
        result["detail"] = "reference %s, target %s" % (ref.get("shape"), got.get("shape"))
        return result

    if gs.get("nan", 0) and not rs.get("nan", 0):
        result["verdict"] = "NAN"
        result["detail"] = "%d NaN values appeared on the target" % gs["nan"]
        return result
    if gs.get("inf", 0) and not rs.get("inf", 0):
        result["verdict"] = "INF"
        result["detail"] = "%d infinities appeared on the target" % gs["inf"]
        return result

    a, b = ref.get("sample") or [], got.get("sample") or []
    n = min(len(a), len(b))
    scale = max(abs(rs.get("absmax", 0.0) or 0.0), abs(gs.get("absmax", 0.0) or 0.0))
    if scale == float("inf"):
        scale = 0.0
    floor = scale * 1e-6 if scale else 0.0

    max_abs = 0.0
    max_rel = 0.0
    worst = None
    disagreeing_infinities = 0
    for i in range(n):
        fa = a[i] == a[i] and abs(a[i]) != float("inf")
        fb = b[i] == b[i] and abs(b[i]) != float("inf")
        if not fa or not fb:
            # Two infinities of the same sign agree.  Anything else here is a
            # disagreement that no finite difference can express.
            if not (fa or fb) and a[i] == b[i]:
                continue
            disagreeing_infinities += 1
            continue
        d = abs(a[i] - b[i])
        if d > max_abs:
            max_abs, worst = d, i
        # A relative error is only reported where there is something to be
        # relative to.
        if abs(a[i]) >= floor or abs(b[i]) >= floor:
            r = _rel(a[i], b[i])
            if r > max_rel:
                max_rel = r

    result["compared"] = n
    if disagreeing_infinities:
        result["verdict"] = "INF"
        result["detail"] = ("%d of %d sampled values are infinite on one device "
                            "and not the other" % (disagreeing_infinities, n))
        result["max_abs_diff"] = max_abs
        return result
    result["max_abs_diff"] = max_abs
    result["max_rel_diff"] = max_rel
    result["scale"] = scale
    scaled = (max_abs / scale) if scale else 0.0
    if scaled != scaled or scaled == float("inf"):
        scaled = float("inf")
    result["scaled_diff"] = scaled
    if worst is not None:
        result["worst_position"] = (ref.get("sample_positions") or list(range(n)))[worst]
        result["worst_values"] = [a[worst], b[worst]]

    for key in ("mean", "std", "absmax"):
        if key in rs and key in gs:
            result["d_" + key] = _rel(rs[key], gs[key])

    if result["bit_identical"]:
        result["verdict"] = "IDENTICAL"
    elif expect == "bitwise":
        result["verdict"] = "NOT_BITWISE"
        result["detail"] = ("this quantity is expected to match exactly, and does "
                            "not (largest difference %.3e)" % max_abs)
    elif result["scaled_diff"] <= tol:
        result["verdict"] = "WITHIN_TOLERANCE"
    else:
        result["verdict"] = "DIVERGENT"
    return result


VERDICT_RANK = {
    "INFO": -1,
    "IDENTICAL": 0,
    "WITHIN_TOLERANCE": 1,
    "NOT_BITWISE": 2,
    "DIVERGENT": 2,
    "NAN": 3,
    "INF": 3,
    "SHAPE": 3,
    "ERROR": 4,
    "MISSING": 4,
}


def worst_verdict(verdicts):
    worst, rank = "IDENTICAL", -1
    for v in verdicts:
        r = VERDICT_RANK.get(v, 2)
        if r > rank:
            worst, rank = v, r
    return worst
