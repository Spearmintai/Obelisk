"""obe - the command line."""

import argparse
import datetime
import json
import os
import sys

from . import __version__


def _p(msg=""):
    print(msg)
    sys.stdout.flush()


# ------------------------------------------------------------------ devices

def cmd_devices(args):
    from . import backend as B
    found = B.available()
    if not found:
        _p("")
        _p("  No accelerator is visible from this interpreter, because PyTorch")
        _p("  is not installed here.")
        _p("")
        _p("  That is fine for most of what obe does. inspect, check, diff and")
        _p("  report all read files and need no hardware. Only record, verify,")
        _p("  dump, run and sample have to be on the machine with the card.")
        _p("")
        return 0
    _p("%d backend(s) available here" % len(found))
    for b in found:
        info = b.info()
        _p("")
        _p("  %s" % info.get("device"))
        for k in ("name", "memory_gb", "capability", "multiprocessors", "backend",
                  "runtime_version", "driver", "torch", "arch", "host",
                  "allow_tf32_matmul"):
            if info.get(k) is not None:
                _p("    %-18s %s" % (k, info[k]))
    return 0


# ------------------------------------------------------------------ inspect

def cmd_inspect(args):
    from . import inspect as I
    result = I.scan(args.path)
    rows = I.summarise(result)
    risk = I.predicted_risk(rows)

    _p("")
    _p("  PRE-FLIGHT INSPECTION   %s" % result["root"])
    _p("  %d files read, %d assumptions found" % (result["files_scanned"], len(result["findings"])))
    _p("")
    _p("  PREDICTED RISK   %s   %s" % (risk["grade"], risk["why"]))
    _p("")
    if rows:
        _p("  %-22s %-9s %5s  %s" % ("rule", "severity", "count", "where"))
        _p("  " + "-" * 74)
        for r in rows:
            _p("  %-22s %-9s %5d  %s" % (r["rule"], r["severity"], r["count"],
                                         ", ".join(r["examples"][:2])))
        _p("")
        _p("  what these mean")
        seen = set()
        for r in rows:
            if r["severity"] in ("critical", "high") and r["meaning"] not in seen:
                seen.add(r["meaning"])
                _p("    %-22s %s" % (r["rule"], r["meaning"]))
    _p("")

    if args.out:
        payload = {"summary": rows, "risk": risk,
                   "files_scanned": result["files_scanned"],
                   "findings": result["findings"] if args.full else result["findings"][:200],
                   "root": result["root"]}
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        _p("  written to %s" % args.out)
    return 0


# ------------------------------------------------------------------ migrate

def cmd_migrate(args):
    from . import migrate as M
    result = M.plan(args.path, target=args.target, apply=args.apply)
    _p("")
    _p("  MIGRATION PLAN   %s  ->  %s" % (args.path, args.target))
    _p("  %d file(s) would change" % result["files_changed"]
       if not args.apply else "  %d file(s) changed" % result["files_changed"])
    for c in result["changes"]:
        _p("    %-42s %d" % (c["change"], c["count"]))
    if result["blockers"]:
        _p("")
        _p("  a mechanical rewrite is not enough here")
        for b in result["blockers"][:20]:
            _p("    %-50s %s" % (b["file"][:50], b["needs"]))
    if result.get("leftovers"):
        _p("")
        _p("  still naming the old backend after the rewrite (%d place(s))"
           % len(result["leftovers"]))
        for lo in result["leftovers"][:12]:
            _p("    %-28s %s" % ("%s:%d" % (lo["file"][:22], lo["line"]), lo["text"][:44]))
    _p("")
    if args.patch:
        with open(args.patch, "w") as fh:
            fh.write(result["patch"])
        _p("  patch written to %s" % args.patch)
    return 0


# ------------------------------------------------------------------ record

def cmd_record(args):
    from . import backend as B, obl, probes, workload

    be = B.pick(args.backend)
    info = be.info()
    _p("")
    _p("  RECORDING REFERENCE on %s  (%s)" % (info.get("name"), be.device))
    _p("")

    _p("  probes")
    probe_results = probes.run_all(be, only=args.only, verbose=True)

    payload = {"probes": probe_results}

    if not args.no_trace:
        _p("  forward trace")
        payload["trace"] = workload.forward_trace(be, dim=args.dim, layers=args.layers)
        _p("    %d layers captured" % len(payload["trace"]["order"]))

    if args.steps:
        _p("  training %d steps" % args.steps)
        payload["train"] = workload.train_trace(be, steps=args.steps,
                                                dim=args.dim, layers=args.layers)
        _p("    loss %.6f -> %.6f" % (payload["train"]["losses"][0],
                                      payload["train"]["losses"][-1]))

    blueprint = obl.new("reference", info, payload)
    path = obl.save(blueprint, args.out)
    size = os.path.getsize(path) / 1024.0
    _p("")
    _p("  blueprint written to %s  (%.0f KB)" % (path, size))
    _p("  carry this file to the target machine; it needs nothing else.")
    _p("")
    return 0


# ------------------------------------------------------------------ verify

def cmd_verify(args):
    from . import backend as B, compare, obl, probes, report, workload

    ref = obl.load(args.ref)
    be = B.pick(args.backend)
    info = be.info()

    _p("")
    _p("  VERIFYING  %s" % info.get("name"))
    _p("  against    %s" % obl.describe(ref))
    _p("")

    ref_payload = ref["payload"]
    _p("  probes")
    got_probes = probes.run_all(be, only=args.only, verbose=True)

    probe_results = compare.compare_probes(ref_payload.get("probes", {}), got_probes, args.tol)
    internal = compare.check_internal(got_probes)

    layer_rows = []
    attribution = []
    if ref_payload.get("trace") and not args.no_trace:
        _p("  forward trace")
        got_trace = workload.forward_trace(be, dim=args.dim, layers=args.layers)
        layer_rows = compare.compare_layers(ref_payload["trace"], got_trace, args.tol)
        attribution = compare.attribute(layer_rows, probe_results)
        _p("    %d layers compared" % len(layer_rows))

    train_cmp = None
    if ref_payload.get("train") and args.steps:
        _p("  training %d steps" % args.steps)
        got_train = workload.train_trace(be, steps=args.steps, dim=args.dim,
                                         layers=args.layers)
        rl, gl = ref_payload["train"]["losses"], got_train["losses"]
        n = min(len(rl), len(gl))
        train_cmp = {
            "steps": n,
            "ref_final": rl[n - 1],
            "got_final": gl[n - 1],
            "loss_gap": abs(rl[n - 1] - gl[n - 1]),
            "first_step_gap": abs(rl[0] - gl[0]) if n else None,
            "ref_losses": rl[:n],
            "got_losses": gl[:n],
            "checkpoints": compare.compare_checkpoints(
                ref_payload["train"], got_train, args.tol),
        }

    grade = compare.grade(probe_results, internal, layer_rows, train_cmp)
    fd = compare.first_divergence(layer_rows) if layer_rows else None

    result = {
        "version": __version__,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "reference": obl.describe(ref),
        "target": "%s on %s (%s, torch %s)" % (
            info.get("name"), info.get("host"), info.get("backend"), info.get("torch")),
        "reference_device": ref.get("device"),
        "target_device": info,
        "probes": probe_results,
        "internal": internal,
        "layers": layer_rows,
        "first_divergence": fd,
        "attribution": attribution,
        "training": train_cmp,
        "grade": grade,
    }

    _p("")
    _p(report.text(result))

    if args.out:
        report.save_json(result, args.out)
        _p("")
        _p("  result written to %s" % args.out)
    if args.html:
        with open(args.html, "w") as fh:
            fh.write(report.html_report(result))
        _p("  report written to %s" % args.html)
    return 0


# ------------------------------------------------------------------ report

def cmd_report(args):
    from . import report
    with open(args.result) as fh:
        result = json.load(fh)
    if args.html:
        with open(args.html, "w") as fh:
            fh.write(report.html_report(result))
        _p("report written to %s" % args.html)
    else:
        _p(report.text(result))
    return 0


# ------------------------------------------------------------------ probes

def cmd_probes(args):
    from . import probes
    _p("")
    _p("  %d probes" % len(probes.REGISTRY))
    _p("")
    layer = None
    for entry in sorted(probes.REGISTRY, key=lambda e: (e["layer"], e["id"])):
        if entry["layer"] != layer:
            layer = entry["layer"]
            _p("  %s" % layer.upper())
        _p("    %-26s %-34s %s" % (entry["id"], entry["title"], entry["severity"]))
    _p("")
    return 0


# ------------------------------------------------------------------ cards

def cmd_cards(args):
    """List the devices this copy of Obelisk already knows about."""
    from . import library

    if args.profile:
        from . import obl
        profile = obl.load(args.profile)
        card = library.from_profile(profile)
        out = args.out or (library.slug(card["device"]) + ".json")
        library.save(card, out)
        size = os.path.getsize(out) / 1024.0
        _p("")
        _p("  card written to %s  (%.1f KB)" % (out, size))
        _p("  %d operators unavailable, %d returning a different answer"
           % (len(card["missing"]), len(card["wrong"])))
        _p("")
        _p("  This is the publishable tier. It says what works and what does")
        _p("  not, and carries no measured values of any kind.")
        _p("")
        return 0

    entries = library.bundled()
    _p("")
    if not entries:
        _p("  No cards are bundled with this copy.")
        _p("  Make one from a profile:  obe cards --profile device.obl")
        _p("")
        return 0
    _p("  %d device(s) measured and shipped with this copy of Obelisk" % len(entries))
    _p("")
    _p("  %-34s %-9s %8s %7s %7s" % ("card", "backend", "missing", "wrong", "ok"))
    _p("  " + "-" * 70)
    for e in entries:
        c = e["card"]
        _p("  %-34s %-9s %8d %7d %7d"
           % (e["name"][:34], (c.get("device") or {}).get("backend", "?"),
              len(c.get("missing", [])), len(c.get("wrong", [])),
              c.get("exact", 0) + c.get("close", 0)))
    _p("")
    _p("  Use one by name:   obe check ./repo --card %s" % entries[0]["name"])
    _p("")
    _p("  Every card records the framework and driver it was measured under.")
    _p("  A card is only true of that configuration.")
    _p("")
    return 0


# ------------------------------------------------------------------ check

def cmd_check(args):
    """Judge a repository against a card that is not here."""
    from . import check as C, obl, shim

    from . import library
    try:
        path, bundled_name = library.resolve(args.card)
    except FileNotFoundError as exc:
        _p("  %s" % exc)
        return 2

    raw = json.load(open(path))
    if raw.get("format") == library.CARD_FORMAT:
        card = raw
        profile = library.as_profile(card)
        notes = library.staleness(card)
    else:
        profile = obl.load(path)
        card = None
        notes = []
        if obl.verify_digest(profile) is False:
            _p("  this profile does not match its own digest; it has been "
               "altered since it was written")
            return 2

    if not os.path.isdir(os.path.expanduser(args.path)):
        _p("  %s is not a directory" % args.path)
        return 2

    result = C.check(args.path, profile, shim.handled())
    if not result["files"]:
        _p("  no Python files found under %s, so there is nothing to check"
           % result["repo"])
        return 2
    grade, why = C.verdict(result)

    _p("")
    _p("=" * 78)
    _p(("  CODE AGAINST CARD").ljust(68) + ("GRADE  %s" % grade))
    _p("=" * 78)
    _p("  code   %s  (%d python files)" % (result["repo"], result["files"]))
    _p("  card   %s  (%s)%s" % (result["card"], result["card_backend"],
                                 "  [bundled]" if bundled_name else ""))
    _p("  taken  %s" % (profile.get("created") or "?"))
    for note in notes:
        _p("  note   %s" % note)
    _p("")
    _p("  %s." % (why[0].upper() + why[1:]))
    _p("")
    _p("  Neither the card nor this code has met the other. The card was")
    _p("  measured on its own, with none of this code present.")
    _p("")

    def show(title, rows, mark):
        if not rows:
            return
        _p("-" * 78)
        _p("  %s" % title)
        for r in rows[:16]:
            tag = "" if not r["weak"] else "  (name match only)"
            fix = {"rewrite": "obe can rewrite it",
                   "fallback": "obe can run it on the CPU",
                   "correct": "obe can replace it"}.get(r["shim"], "no substitute")
            _p("   %s  %-26s %-9s %-24s %d call site%s%s"
               % (mark, r["op"], r["status"], fix, r["count"],
                  "" if r["count"] == 1 else "s", tag))
            for f, line in r["sites"][:3]:
                _p("        %s:%d" % (f, line))
        if len(rows) > 16:
            _p("      and %d more" % (len(rows) - 16))
        _p("")

    show("WILL NOT RUN", result["blocking"], "x")
    show("RUNS AND RETURNS THE WRONG ANSWER", result["wrong"], "!")
    show("RUNS, WITH A DIFFERENT RESULT", result["risky"], "~")

    if result["dtypes"]:
        _p("-" * 78)
        _p("  DATA TYPES THIS CARD CANNOT COMPUTE WITH")
        for row in result["dtypes"]:
            _p("   %-12s %s" % (row["dtype"], ", ".join(row["unsupported"])))
        _p("")

    shimmable = len([r for r in result["blocking"] if r["shim"]])
    _p("-" * 78)
    _p("  %d operation(s) verified to behave the same" % result["confirmed_ok"])
    if result["blocking"]:
        _p("  %d of %d blocking operations can be substituted by obe run"
           % (shimmable, len(result["blocking"])))
    _p("  card profile digest %s" % (result["card_digest"] or "?"))
    _p("=" * 78)

    if args.out:
        result["grade"] = {"grade": grade, "why": why}
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        _p("")
        _p("  written to %s" % args.out)
    return 0


# ------------------------------------------------------------------ run

def cmd_run(args):
    """Run a program with the gaps filled in, and report what was filled."""
    import runpy
    import sys as _sys
    from . import backend as B, shim

    be = B.pick(args.backend)
    _p("")
    _p("  OBELISK RUN  on %s" % be.info().get("name"))
    _p("")
    _p("  checking which operators need standing in for")
    ledger = shim.activate(be, verbose=True)
    rows = ledger.rows()
    _p("")
    _p("  %d operator(s) patched: %d rewritten, %d corrected, %d sent to the CPU"
       % (len(rows),
          len([r for r in rows if r["strategy"] == "rewrite"]),
          len([r for r in rows if r["strategy"] == "correct"]),
          len([r for r in rows if r["strategy"] == "fallback"])))
    _p("")
    _p("-" * 78)

    argv = [args.script] + list(args.args or [])
    old_argv = _sys.argv
    _sys.argv = argv
    status = 0
    try:
        runpy.run_path(args.script, run_name="__main__")
    except SystemExit as exc:
        status = exc.code if isinstance(exc.code, int) else 0
    except Exception as exc:
        _p("")
        _p("  the program raised: %s: %s" % (type(exc).__name__, exc))
        status = 1
        if os.environ.get("OBE_TRACEBACK"):
            raise
    finally:
        _sys.argv = old_argv

    _p("-" * 78)
    _p("")
    _p("  LEDGER: what did not happen the way you wrote it")
    _p("")
    used = [r for r in ledger.rows() if r["calls"]]
    if not used:
        _p("   Nothing. Every operator this program called was the device's own.")
    else:
        _p("   %-22s %-9s %8s %9s %13s"
           % ("operator", "how", "calls", "seconds", "accuracy"))
        for r in sorted(used, key=lambda r: -r["calls"]):
            acc = r.get("accuracy")
            shown = ("%.1e" % acc["error"]) if acc else "exact"
            _p("   %-22s %-9s %8d %9.3f %13s"
               % (r["op"], r["strategy"], r["calls"], r["seconds"], shown))
        degraded = [r for r in used if r.get("accuracy")
                    and r["accuracy"]["baseline"] > 0
                    and r["accuracy"]["error"] > 20 * r["accuracy"]["baseline"]]
        for r in degraded:
            a = r["accuracy"]
            _p("")
            _p("   %s is accurate to %.1e here, where a proper kernel reaches "
               "%.1e. The gap is widest %s."
               % (r["op"], a["error"], a["baseline"], a["regime"]))
        s = ledger.summary()
        _p("")
        _p("   %d call(s) through %d substituted operator(s), %.3f s total"
           % (s["calls"], s["exercised"], s["seconds"]))
        if s["fallback_seconds"] > 0:
            _p("   %.3f s of that was spent moving data to the CPU and back"
               % s["fallback_seconds"])
        corrected = [r for r in used if r["strategy"] == "correct"]
        for r in corrected:
            _p("   %s was called %d time(s); the device kernel is wrong, so the "
               "result you got is not the result the device would have given"
               % (r["op"], r["calls"]))
    _p("")

    if args.ledger:
        with open(args.ledger, "w") as fh:
            json.dump({"device": be.info(), "summary": ledger.summary(),
                       "operators": ledger.rows()}, fh, indent=1)
        _p("  ledger written to %s" % args.ledger)
    return status


# ------------------------------------------------------------------ dump

def cmd_dump(args):
    """Record everything measurable about one device."""
    from . import backend as B, coverage, obl, probes, workload

    be = B.pick(args.backend)
    info = be.info()
    _p("")
    _p("  PROFILING  %s  (%s)" % (info.get("name"), be.device))
    _p("")

    payload = {}

    _p("  dtypes")
    payload["dtypes"] = coverage.dtype_matrix(be)
    usable = [k for k, v in payload["dtypes"].items() if v.get("matmul")]
    _p("    %d of %d dtypes can do arithmetic here" %
       (len(usable), len(payload["dtypes"])))

    _p("  operators")

    def progress(i, total):
        _p("    %d / %d" % (i, total))

    cov = coverage.sweep(be, limit=args.limit, verbose=args.verbose,
                         progress=progress if not args.quiet else None)
    payload["coverage"] = cov
    c = cov["counts"]
    _p("    %d callables discovered, %d exercised, %d not applicable"
       % (cov["candidates"], c["tested"], c["skipped"]))
    _p("    %d exact   %d close   %d different   %d missing   %d failing"
       % (c["exact"], c["close"], c["mismatch"], c["missing"], c["error"]))

    if not args.no_probes:
        _p("  probes")
        payload["probes"] = probes.run_all(be, verbose=False)
        _p("    %d assumptions measured" % len(payload["probes"]))

    if not args.no_trace:
        _p("  forward trace")
        payload["trace"] = workload.forward_trace(be)
        _p("    %d layers captured" % len(payload["trace"]["order"]))

    blueprint = obl.new("profile", info, payload)
    path = obl.save(blueprint, args.out)
    size = os.path.getsize(path) / 1024.0
    _p("")
    _p("  profile written to %s  (%.0f KB)" % (path, size))
    _p("  digest %s" % blueprint["digest"])
    _p("")
    return 0


# ------------------------------------------------------------------ diff

def cmd_diff(args):
    """Compare two recorded profiles with no hardware present."""
    from . import obl, profile, report

    ref = obl.load(args.reference)
    got = obl.load(args.target)

    for name, bp in (("reference", ref), ("target", got)):
        ok = obl.verify_digest(bp)
        if ok is False:
            _p("  the %s profile does not match its own digest; it has been "
               "altered since it was written" % name)
            return 2

    result = profile.compare_profiles(ref, got, args.tol)
    _p("")
    _p(report.profile_text(result))

    if args.out:
        report.save_json(result, args.out)
        _p("")
        _p("  written to %s" % args.out)
    return 0


# ------------------------------------------------------------------ sample

def cmd_sample(args):
    """The smallest possible demonstration: identical code, identical seed."""
    import torch
    from . import backend as B

    be = B.pick(args.backend)
    dev = be.device
    info = be.info()
    _p("")
    _p("  device   %s" % info.get("name"))
    _p("  backend  %s" % info.get("backend"))
    _p("  torch    %s" % info.get("torch"))
    _p("")

    def show(label, values):
        _p("  %-22s %s" % (label, "  ".join("%+.8f" % v for v in values)))

    torch.manual_seed(args.seed)
    show("rand (uniform)", [float(v) for v in torch.rand(4, device=dev)])
    torch.manual_seed(args.seed)
    show("randn (normal)", [float(v) for v in torch.randn(4, device=dev)])
    torch.manual_seed(args.seed)
    layer = torch.nn.Linear(8, 4).to(dev)
    show("Linear init weights", [float(v) for v in layer.weight[0][:4]])

    x = torch.arange(1, 5, dtype=torch.float32, device=dev)
    for name in ("log1p", "erf", "expm1", "digamma", "i0"):
        fn = getattr(torch, name, None)
        if fn is None:
            continue
        try:
            show(name, [float(v) for v in fn(x)])
        except Exception as exc:
            _p("  %-22s %s" % (name, ("%s: %s" % (type(exc).__name__,
               str(exc).split(chr(10))[0]))[:52]))
    _p("")
    _p("  Run this on two machines and read the two outputs side by side.")
    _p("")
    return 0


# ------------------------------------------------------------------ main

def build_parser():
    ap = argparse.ArgumentParser(
        prog="obe",
        description="Verify that code runs correctly on a different accelerator, "
                    "not just that it runs.")
    ap.add_argument("--version", action="version", version="obelisk %s" % __version__)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("devices", help="list the accelerators visible here")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("probes", help="list the assumptions Obelisk knows how to test")
    p.set_defaults(fn=cmd_probes)

    p = sub.add_parser("inspect", help="read a repository and predict its risk")
    p.add_argument("path")
    p.add_argument("--out", help="write the findings as JSON")
    p.add_argument("--full", action="store_true", help="keep every finding, not the first 200")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("migrate", help="rewrite a repository for another backend")
    p.add_argument("path")
    p.add_argument("--target", default="musa", choices=["musa", "cuda"])
    p.add_argument("--apply", action="store_true", help="write the changes to disk")
    p.add_argument("--patch", help="write the unified diff here")
    p.set_defaults(fn=cmd_migrate)

    common = dict(dim=256, layers=4)
    p = sub.add_parser("record", help="measure this device and write a reference blueprint")
    p.add_argument("--backend", default="auto")
    p.add_argument("--out", default="reference.obl")
    p.add_argument("--only", nargs="*", help="only these probe ids or prefixes")
    p.add_argument("--steps", type=int, default=50, help="training steps to record (0 to skip)")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--dim", type=int, default=common["dim"])
    p.add_argument("--layers", type=int, default=common["layers"])
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("verify", help="compare this device against a reference blueprint")
    p.add_argument("--ref", required=True)
    p.add_argument("--backend", default="auto")
    p.add_argument("--only", nargs="*")
    p.add_argument("--tol", type=float, default=None, help="override the dtype tolerance")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--dim", type=int, default=common["dim"])
    p.add_argument("--layers", type=int, default=common["layers"])
    p.add_argument("--out", help="write the full result as JSON")
    p.add_argument("--html", help="write the report as a web page")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("cards", help="list the devices Obelisk already knows about")
    p.add_argument("--profile", help="reduce this profile to a publishable card")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_cards)

    p = sub.add_parser("check", help="judge a repository against a card profiled elsewhere")
    p.add_argument("path")
    p.add_argument("--card", required=True,
                   help="a profile from obe dump, or the name of a bundled card")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="run a program with missing operators filled in")
    p.add_argument("script")
    p.add_argument("args", nargs="*")
    p.add_argument("--backend", default="auto")
    p.add_argument("--ledger", help="write the ledger as JSON")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("dump", help="record everything measurable about this device")
    p.add_argument("--backend", default="auto")
    p.add_argument("--out", default="device.obl")
    p.add_argument("--limit", type=int, default=None, help="stop after N operators")
    p.add_argument("--no-probes", action="store_true")
    p.add_argument("--no-trace", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("diff", help="compare two recorded profiles, no hardware needed")
    p.add_argument("reference")
    p.add_argument("target")
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("sample", help="print a few numbers; run it on two machines")
    p.add_argument("--backend", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_sample)

    p = sub.add_parser("report", help="render a saved result")
    p.add_argument("result")
    p.add_argument("--html")
    p.set_defaults(fn=cmd_report)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _p("")
        _p("  %s: %s" % (type(exc).__name__, exc))
        if os.environ.get("OBE_TRACEBACK"):
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
