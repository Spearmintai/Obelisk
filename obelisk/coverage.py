"""Operator coverage: what this device can actually do.

The fifteen probes ask whether a device holds the assumptions a program makes.
This asks something broader and more tedious: of everything PyTorch exposes, what
works here, what is missing, and what returns a different answer.

The method rests on one decision.  The CPU is the oracle for whether a call is
*valid*, and the device is the subject under test.  Every case is run on the CPU
first; if it fails there, the harness called it wrong and the case is discarded
rather than counted against the device.  Only a call that the CPU accepts and the
device refuses is a finding.  That rule is what separates a coverage report from a
list of the author's mistakes.
"""

import math
import re

MISSING_PATTERNS = re.compile(
    r"not implemented|could not run|no kernel|unsupported|not supported|"
    r"doesn't exist for this backend|NotImplementedError", re.I)

# Anything that touches the outside world, consumes randomness, or names a
# specific backend is not a question about numerical behaviour.
SKIP = re.compile(
    r"^_|seed|random|rand|normal_|bernoulli|multinomial|poisson|"
    r"save|load|print|compile|jit|trace|script|profiler|distributed|"
    r"cuda|musa|xpu|mps|hip|device|stream|event|memory|pin|share|"
    r"set_|get_|is_|has_|enable|disable|use_|init|quantize|fake_quant|"
    r"sparse|nested|masked_|_foreach|future|wait|fork|exit|hook|"
    r"storage|untyped|from_|as_|to_|numpy|item|tolist|backward|grad|"
    r"requires|detach|register|parameter|module|autograd|func|vmap|"
    r"export|onnx|package|fx|library|custom|op_|dispatch|lobpcg|"
    r"searchsorted|bucketize|histogramdd|unique_consecutive|segment")


def _templates(torch):
    """Candidate argument shapes, tried in order until the CPU accepts one.

    The order matters: narrower templates come first so that an operator which
    accepts several signatures is exercised in its most ordinary form.
    """
    g = torch.Generator().manual_seed(4242)
    small = torch.randn(8, 8, generator=g)
    pos = small.abs() + 0.5
    unit = torch.rand(8, 8, generator=g) * 0.8 + 0.1
    square = torch.randn(6, 6, generator=g)
    spd = square @ square.t() + 6 * torch.eye(6)
    ints = torch.randint(0, 5, (8, 8), generator=g)
    vec = torch.randn(16, generator=g)
    batch = torch.randn(2, 3, 8, 8, generator=g)

    return [
        ("unary", lambda: (small.clone(),), {}),
        ("unary_positive", lambda: (pos.clone(),), {}),
        ("unary_unit", lambda: (unit.clone(),), {}),
        ("binary", lambda: (small.clone(), small.clone() + 1.0), {}),
        ("scalar", lambda: (small.clone(), 2.0), {}),
        ("reduce_dim", lambda: (small.clone(),), {"dim": 1}),
        ("matrix", lambda: (square.clone(),), {}),
        ("spd_matrix", lambda: (spd.clone(),), {}),
        ("vector", lambda: (vec.clone(),), {}),
        ("batched", lambda: (batch.clone(),), {}),
        ("integer", lambda: (ints.clone(),), {}),
        ("pair_int", lambda: (ints.clone(), ints.clone() + 1), {}),
        ("unary_k", lambda: (small.clone(), 3), {}),
    ]


def _collect(torch):
    """Every public callable worth asking about, from torch and torch.nn.functional."""
    cases = []
    for ns_name, ns in (("torch", torch), ("F", torch.nn.functional)):
        for name in dir(ns):
            if SKIP.search(name):
                continue
            fn = getattr(ns, name, None)
            if not callable(fn) or isinstance(fn, type):
                continue
            cases.append(("%s.%s" % (ns_name, name), fn))
    return sorted(set(cases), key=lambda c: c[0])


def _move(args, kwargs, dev):
    import torch
    a = tuple(x.to(dev) if hasattr(x, "to") and hasattr(x, "dtype") else x for x in args)
    k = {kk: (v.to(dev) if hasattr(v, "to") and hasattr(v, "dtype") else v)
         for kk, v in kwargs.items()}
    return a, k


def _flatten(out, torch):
    """Reduce whatever an operator returned to a list of tensors."""
    if hasattr(out, "dtype") and hasattr(out, "shape"):
        return [out]
    if isinstance(out, (tuple, list)):
        found = []
        for item in out:
            found.extend(_flatten(item, torch))
        return found
    if hasattr(out, "values") and hasattr(out, "indices"):
        return _flatten(out.values, torch) + _flatten(out.indices, torch)
    return []


TOL = {"float32": 1e-4, "float64": 1e-10, "float16": 2e-2, "bfloat16": 6e-2}


def _agreement(ref_tensors, got_tensors, torch):
    """How far apart two results are, relative to the scale of the reference."""
    if len(ref_tensors) != len(got_tensors):
        return "SHAPE", 0.0
    worst = 0.0
    exact = True
    for r, g in zip(ref_tensors, got_tensors):
        if tuple(r.shape) != tuple(g.shape):
            return "SHAPE", 0.0
        rf = r.detach().to("cpu").to(torch.float64).reshape(-1)
        gf = g.detach().to("cpu").to(torch.float64).reshape(-1)
        if rf.numel() == 0:
            continue
        finite = torch.isfinite(rf) & torch.isfinite(gf)
        if not bool(finite.all()):
            if not bool((torch.isfinite(rf) == torch.isfinite(gf)).all()):
                return "NONFINITE", float("inf")
            rf, gf = rf[finite], gf[finite]
            if rf.numel() == 0:
                continue
        diff = float((rf - gf).abs().max())
        if diff != 0.0:
            exact = False
        scale = float(rf.abs().max())
        worst = max(worst, diff / scale if scale > 0 else diff)
    if exact:
        return "EXACT", 0.0
    return "NUMERIC", worst


def sweep(backend, dtypes=("float32",), limit=None, verbose=False, progress=None):
    """Run every discoverable operator on the CPU and on the device."""
    import torch

    dev = backend.device
    cases = _collect(torch)
    if limit:
        cases = cases[:limit]
    templates = _templates(torch)

    results = []
    counts = {"tested": 0, "skipped": 0, "missing": 0, "error": 0,
              "mismatch": 0, "exact": 0, "close": 0}

    for index, (name, fn) in enumerate(cases):
        if progress and index % 25 == 0:
            progress(index, len(cases))

        chosen = None
        ref_out = None
        for tname, make, kw in templates:
            try:
                args = make()
                ref = fn(*args, **kw)
                tensors = _flatten(ref, torch)
                if not tensors:
                    continue
                if not any(t.is_floating_point() or t.dtype in
                           (torch.int64, torch.int32, torch.bool) for t in tensors):
                    continue
                chosen = (tname, make, kw)
                ref_out = tensors
                break
            except Exception:
                continue

        if chosen is None:
            counts["skipped"] += 1
            continue

        tname, make, kw = chosen
        counts["tested"] += 1
        record = {"op": name, "template": tname}

        try:
            args, kwargs = _move(make(), kw, dev)
            got = fn(*args, **kwargs)
            backend.synchronize()
            got_tensors = _flatten(got, torch)
        except Exception as exc:
            text = "%s: %s" % (type(exc).__name__, str(exc).split("\n")[0])
            if MISSING_PATTERNS.search(text):
                record["status"] = "MISSING"
                counts["missing"] += 1
            else:
                record["status"] = "ERROR"
                counts["error"] += 1
            record["detail"] = text[:200]
            results.append(record)
            if verbose:
                print("    %-40s %s" % (name, record["status"]))
            continue

        kind, worst = _agreement(ref_out, got_tensors, torch)
        dtype = str(ref_out[0].dtype).replace("torch.", "")
        tol = TOL.get(dtype, 1e-4)
        if kind == "SHAPE":
            record["status"] = "SHAPE"
            counts["mismatch"] += 1
        elif kind == "NONFINITE":
            record["status"] = "MISMATCH"
            record["detail"] = "infinities or NaN appear on only one side"
            counts["mismatch"] += 1
        elif kind == "EXACT":
            record["status"] = "EXACT"
            counts["exact"] += 1
        elif worst <= tol:
            record["status"] = "CLOSE"
            record["error"] = worst
            counts["close"] += 1
        else:
            record["status"] = "MISMATCH"
            record["error"] = worst
            counts["mismatch"] += 1
        results.append(record)
        if verbose and record["status"] in ("MISMATCH", "SHAPE"):
            print("    %-40s %-10s %.2e" % (name, record["status"],
                                            record.get("error") or 0.0))

    return {"counts": counts, "operators": results,
            "candidates": len(cases), "dtypes": list(dtypes)}


def dtype_matrix(backend):
    """Which dtypes the device will actually compute with, not merely store."""
    import torch
    dev = backend.device
    out = {}
    names = ["float32", "float64", "float16", "bfloat16", "int64", "int32",
             "int16", "int8", "uint8", "bool", "complex64"]
    for name in names:
        dt = getattr(torch, name, None)
        if dt is None:
            continue
        row = {}
        try:
            x = torch.ones(16, 16, dtype=dt, device=dev)
            row["allocate"] = True
        except Exception as exc:
            out[name] = {"allocate": False,
                         "detail": str(exc).split("\n")[0][:120]}
            continue
        for op, call in (
            ("add", lambda t: t + t),
            ("mul", lambda t: t * t),
            ("matmul", lambda t: t @ t),
            ("sum", lambda t: t.sum()),
        ):
            try:
                call(x)
                backend.synchronize()
                row[op] = True
            except Exception:
                row[op] = False
        out[name] = row
    return out
