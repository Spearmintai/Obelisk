"""Comparison, attribution and grading.

Comparison answers "where do the two devices disagree".  Attribution answers the
question a comparison on its own never can: *why*.  A layer that disagrees is a
symptom; the probe that disagrees in the same way is the cause.
"""

from .fingerprint import compare_fingerprints, worst_verdict, VERDICT_RANK


def _short(msg, limit=220):
    """Error text from a dispatcher can run to hundreds of lines.  Keep the part
    that names the problem."""
    msg = str(msg or "")
    first = msg.split("\n", 1)[0].strip()
    return first[:limit] + ("..." if len(first) > limit else "")

# Which probes can explain a disagreement in a layer of a given kind.  The order
# within each list is the order in which causes are considered.
ATTRIBUTION = [
    ("ln", ["lib.normalisation", "acc.reduction_order"]),
    ("norm", ["lib.normalisation", "acc.reduction_order"]),
    ("qkv", ["acc.fp32_matmul", "acc.reduction_order", "acc.fp16_accumulate"]),
    ("proj", ["acc.fp32_matmul", "acc.reduction_order"]),
    ("fc1", ["acc.fp32_matmul", "acc.reduction_order"]),
    ("fc2", ["acc.fp32_matmul", "acc.reduction_order", "lib.transcendental"]),
    ("head", ["acc.fp32_matmul", "acc.reduction_order"]),
    ("emb", ["mem.layout"]),
    ("att", ["net.attention", "lib.softmax"]),
    ("__output__", ["acc.fp32_matmul", "net.attention", "lib.softmax",
                    "acc.reduction_order"]),
]

# Invariants that must hold on a single device, with no reference at all.
INTERNAL = [
    ("ord.nondeterminism", "run1", "run2",
     "the same matmul, run twice on this device, gave two different answers"),
    ("rng.device_stream", "randn", "randn_reseeded",
     "re-seeding the generator did not reproduce the same numbers"),
    ("net.attention", "attention_manual", "attention_fused",
     "the fused attention kernel disagrees with the unfused computation"),
]


def compare_probes(ref, got, tol=None):
    """Compare two probe runs probe by probe, tensor by tensor."""
    out = {}
    for pid, rrec in ref.items():
        grec = got.get(pid)
        meta = {k: rrec.get(k) for k in ("title", "layer", "symptom", "severity",
                                         "expect", "cross_device")}
        if rrec.get("cross_device") is False:
            # Only meaningful on one device at a time; the internal check covers it.
            out[pid] = dict(meta, verdict="INTERNAL_ONLY",
                            detail="checked on the target alone, not against the reference")
            continue
        if grec is None:
            out[pid] = dict(meta, verdict="MISSING",
                            detail="the target did not run this probe")
            continue
        if rrec.get("status") != "ok" or grec.get("status") != "ok":
            detail = _short(grec.get("error") or rrec.get("error") or "probe failed")
            verdict = "ERROR"
            if rrec.get("status") == "ok" and grec.get("status") != "ok":
                detail = "unsupported or failing on the target: " + _short(grec.get("error"))
            out[pid] = dict(meta, verdict=verdict, detail=detail)
            continue
        tensors = {}
        for name, rf in rrec["tensors"].items():
            gf = grec["tensors"].get(name)
            if gf is None:
                tensors[name] = {"verdict": "MISSING"}
                continue
            tensors[name] = compare_fingerprints(
                rf, gf, tol,
                expect=rrec.get("expect", "numeric"),
                info=name.endswith("_info"))
        verdict = worst_verdict([t.get("verdict", "ERROR") for t in tensors.values()])
        worst_name = None
        worst_rank = -1
        for name, t in tensors.items():
            r = VERDICT_RANK.get(t.get("verdict"), 2)
            if r > worst_rank:
                worst_rank, worst_name = r, name
        rec = dict(meta, verdict=verdict, tensors=tensors, worst_tensor=worst_name)
        if worst_name and tensors[worst_name].get("scaled_diff") is not None:
            rec["scaled_diff"] = tensors[worst_name]["scaled_diff"]
            rec["max_abs_diff"] = tensors[worst_name].get("max_abs_diff")
        info = {k: v for k, v in tensors.items() if v.get("verdict") == "INFO"}
        if info:
            rec["measurements"] = info
        out[pid] = rec
    return out


def check_internal(run):
    """Check the invariants that need no reference device."""
    findings = []
    for pid, a, b, message in INTERNAL:
        rec = run.get(pid)
        if not rec or rec.get("status") != "ok":
            continue
        fa, fb = rec["tensors"].get(a), rec["tensors"].get(b)
        if not fa or not fb:
            continue
        cmp = compare_fingerprints(fa, fb)
        if cmp["verdict"] in ("DIVERGENT", "NAN", "INF", "SHAPE"):
            findings.append({
                "probe": pid,
                "severity": "critical",
                "message": message,
                "max_rel_diff": cmp.get("max_rel_diff"),
                "verdict": cmp["verdict"],
            })
        elif not cmp["bit_identical"] and pid == "ord.nondeterminism":
            findings.append({
                "probe": pid,
                "severity": "high",
                "message": message + " (within tolerance, but not bit-identical)",
                "max_rel_diff": cmp.get("max_rel_diff"),
                "verdict": cmp["verdict"],
            })
    return findings


def compare_layers(ref_trace, got_trace, tol=None):
    """Compare a forward pass layer by layer, in execution order."""
    rows = []
    ref_layers = ref_trace.get("layers", {})
    got_layers = got_trace.get("layers", {})
    for name in ref_trace.get("order", []):
        rf = ref_layers.get(name)
        gf = got_layers.get(name)
        if rf is None:
            continue
        if gf is None:
            rows.append({"layer": name, "verdict": "MISSING"})
            continue
        cmp = compare_fingerprints(rf, gf, tol)
        cmp["layer"] = name
        rows.append(cmp)
    return rows


def attribute(layer_rows, probe_results):
    """For each diverging layer, name the probes that could explain it and that
    are themselves diverging."""
    broken = {pid for pid, rec in probe_results.items()
              if rec.get("verdict") in ("DIVERGENT", "NAN", "INF", "ERROR", "SHAPE",
                                        "NOT_BITWISE")}
    out = []
    for row in layer_rows:
        if row.get("verdict") in ("IDENTICAL", "WITHIN_TOLERANCE", None):
            continue
        name = row["layer"]
        candidates = []
        for key, pids in ATTRIBUTION:
            if key in name:
                for pid in pids:
                    if pid in broken and pid not in candidates:
                        candidates.append(pid)
        out.append({
            "layer": name,
            "verdict": row["verdict"],
            "scaled_diff": row.get("scaled_diff"),
            "causes": [{"probe": pid,
                        "title": probe_results[pid].get("title"),
                        "symptom": probe_results[pid].get("symptom")}
                       for pid in candidates],
        })
    return out


def first_divergence(layer_rows):
    """The earliest layer at which the two devices stop agreeing.  Everything
    after it is downstream of this one."""
    for row in layer_rows:
        if row.get("verdict") not in ("IDENTICAL", "WITHIN_TOLERANCE", None):
            return row
    for row in layer_rows:
        if row.get("verdict") == "WITHIN_TOLERANCE" and not row.get("bit_identical"):
            return row
    return None


def compare_checkpoints(ref_train, got_train, tol=None):
    """Compare the model parameters themselves at the recorded steps."""
    rows = []
    rc = (ref_train or {}).get("checkpoints") or {}
    gc = (got_train or {}).get("checkpoints") or {}
    for key in sorted(rc):
        if key not in gc:
            continue
        cmp = compare_fingerprints(rc[key], gc[key], tol)
        cmp["checkpoint"] = key
        rows.append(cmp)
    return rows


GRADES = ["A", "B", "C", "D", "E"]


def grade(probe_results, internal_findings, layer_rows=None, train=None):
    """Turn the evidence into one letter, and say what drove it.

    A  bit-identical throughout
    B  differs, but every difference is inside the tolerance for its dtype
    C  at least one assumption is broken; results are usable only with care
    D  a critical assumption is broken, or an operation is missing
    E  the device does not agree with itself
    """
    reasons = []
    worst = "A"

    def bump(g, why):
        nonlocal worst
        if GRADES.index(g) > GRADES.index(worst):
            worst = g
        reasons.append({"grade": g, "reason": why})

    for f in internal_findings:
        bump("E", f["message"])

    for pid, rec in (probe_results or {}).items():
        v = rec.get("verdict")
        sev = rec.get("severity", "high")
        if v in ("INTERNAL_ONLY", "INFO"):
            continue
        if v in ("ERROR", "MISSING"):
            bump("D", "%s: %s" % (rec.get("title", pid), rec.get("detail", "failed")))
        elif v in ("NAN", "INF", "SHAPE"):
            bump("D", "%s: %s" % (rec.get("title", pid), rec.get("detail", v)))
        elif v == "NOT_BITWISE":
            bump("C", "%s does not match exactly, and this is a quantity that "
                      "should" % rec.get("title", pid))
        elif v == "DIVERGENT":
            bump("D" if sev == "critical" else "C",
                 "%s differs by %.2e of its own scale"
                 % (rec.get("title", pid), rec.get("scaled_diff") or 0.0))
        elif v == "WITHIN_TOLERANCE":
            bump("B", "%s differs within tolerance" % rec.get("title", pid))

    if layer_rows:
        fd = first_divergence(layer_rows)
        if fd and fd.get("verdict") == "DIVERGENT":
            bump("C", "the forward pass first diverges at %s" % fd["layer"])

    if train and train.get("loss_gap") is not None:
        gap = train["loss_gap"]
        if gap > 1e-2:
            bump("D", "training loss separates by %.4f over %d steps"
                 % (gap, train.get("steps", 0)))
        elif gap > 1e-4:
            bump("C", "training loss separates by %.2e over %d steps"
                 % (gap, train.get("steps", 0)))

    return {"grade": worst, "reasons": reasons}


SUMMARY = {
    "A": "Bit-identical. The target reproduces the reference exactly.",
    "B": "Equivalent within tolerance. Differences exist but none exceed what the "
         "dtype allows.",
    "C": "Runs, but does not agree. At least one hardware assumption does not hold "
         "on the target.",
    "D": "Not safe to deploy as-is. A critical assumption is broken or an "
         "operation is missing.",
    "E": "Unusable as a reference. The device does not reproduce its own results.",
}
