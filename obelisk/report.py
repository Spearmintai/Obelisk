"""Rendering a result: for a terminal, and for a person who has to sign it."""

import html
import json

from .compare import SUMMARY

GRADE_COLOR = {"A": "#1d7a4c", "B": "#3f7a3f", "C": "#a86b1a", "D": "#b3401f", "E": "#8f1d1d"}


def _verdict_mark(v):
    return {
        "IDENTICAL": "=",
        "WITHIN_TOLERANCE": "~",
        "NOT_BITWISE": "!",
        "INTERNAL_ONLY": ".",
        "INFO": "i",
        "DIVERGENT": "!",
        "NAN": "!",
        "INF": "!",
        "SHAPE": "!",
        "ERROR": "x",
        "MISSING": "?",
    }.get(v, "?")


def text(result, width=78):
    """A report that fits in a terminal."""
    L = []
    g = result["grade"]["grade"]
    ref, tgt = result["reference"], result["target"]

    L.append("=" * width)
    L.append("  ENVIRONMENT ALIGNMENT REPORT".ljust(width - 10) + ("GRADE  %s" % g))
    L.append("=" * width)
    L.append("  reference   %s" % ref)
    L.append("  target      %s" % tgt)
    L.append("")
    L.append("  " + SUMMARY.get(g, ""))
    L.append("")

    internal = result.get("internal") or []
    if internal:
        L.append("-" * width)
        L.append("  THE TARGET DOES NOT AGREE WITH ITSELF")
        for f in internal:
            L.append("   !  %s" % f["message"])
            if f.get("max_rel_diff"):
                L.append("      relative difference %.3e" % f["max_rel_diff"])
        L.append("")

    probes = result.get("probes") or {}
    if probes:
        L.append("-" * width)
        L.append("  ASSUMPTIONS")
        order = {"DIVERGENT": 0, "ERROR": 0, "NAN": 0, "INF": 0, "SHAPE": 0,
                 "NOT_BITWISE": 1, "MISSING": 1, "WITHIN_TOLERANCE": 2,
                 "IDENTICAL": 3, "INTERNAL_ONLY": 4}
        for pid, rec in sorted(probes.items(),
                               key=lambda kv: (order.get(kv[1].get("verdict"), 9), kv[0])):
            v = rec.get("verdict", "?")
            line = "   %s  %-26s %-20s %s" % (
                _verdict_mark(v), pid, rec.get("title", "")[:20], v)
            if rec.get("scaled_diff") is not None:
                line += "  %.2e" % rec["scaled_diff"]
            L.append(line)
            if rec.get("measurements"):
                for mname, m in rec["measurements"].items():
                    L.append("        %-30s reference %-12.4g target %.4g"
                             % (mname.replace("_info", ""),
                                m.get("reference_value") or 0.0,
                                m.get("target_value") or 0.0))
            if v in ("DIVERGENT", "ERROR", "NAN", "INF", "SHAPE", "NOT_BITWISE"):
                for seg in _wrap(rec.get("detail") or rec.get("symptom") or "", width - 10):
                    L.append("        %s" % seg)
        L.append("")

    layers = result.get("layers") or []
    if layers:
        fd = result.get("first_divergence")
        L.append("-" * width)
        L.append("  FORWARD PASS  (%d layers compared)" % len(layers))
        diverging = [r for r in layers if r.get("verdict") == "DIVERGENT"]
        if diverging:
            L.append("   first divergence beyond tolerance at  %s" % diverging[0]["layer"])
        elif fd:
            L.append("   no layer diverges beyond tolerance")
            L.append("   earliest layer that is not bit-identical  %s" % fd["layer"])
        else:
            L.append("   every layer is bit-identical")
        shown = [r for r in layers if r.get("verdict") not in ("IDENTICAL",)][:12]
        for r in shown:
            L.append("   %s  %-34s %-18s %s" % (
                _verdict_mark(r.get("verdict")), r.get("layer", "")[:34],
                r.get("verdict", ""),
                ("%.2e" % r["scaled_diff"]) if r.get("scaled_diff") is not None else ""))
        L.append("")

    attribution = result.get("attribution") or []
    if not attribution and (result.get("layers") or []):
        L.append("-" * width)
        L.append("  WHY")
        L.append("   No layer of this workload diverged beyond tolerance, so there is")
        L.append("   nothing here to attribute. The assumptions listed above are still")
        L.append("   broken; this workload simply does not depend on them.")
        L.append("")
    if attribution:
        L.append("-" * width)
        L.append("  WHY")
        for a in attribution[:6]:
            L.append("   %s diverges" % a["layer"])
            for c in a["causes"]:
                L.append("      caused by  %s" % c["title"])
                for seg in _wrap(c["symptom"] or "", width - 12):
                    L.append("         %s" % seg)
            if not a["causes"]:
                L.append("      no probe explains this one; it is specific to the workload")
        L.append("")

    train = result.get("training")
    if train and train.get("loss_gap") is not None:
        L.append("-" * width)
        L.append("  TRAINING  (%d steps)" % train.get("steps", 0))
        L.append("   final loss   reference %.6f   target %.6f" %
                 (train["ref_final"], train["got_final"]))
        L.append("   separation   %.3e" % train["loss_gap"])
        if train.get("first_step_gap") is not None:
            L.append("   separation after a single step  %.3e" % train["first_step_gap"])
        cps = train.get("checkpoints") or []
        if cps:
            L.append("")
            L.append("   model parameters at each recorded step")
            for c in cps:
                L.append("     %-10s %-18s %s" % (
                    c.get("checkpoint"), c.get("verdict"),
                    ("%.2e" % c["scaled_diff"]) if c.get("scaled_diff") else ""))
        L.append("")

    L.append("-" * width)
    L.append("  WHAT DROVE THE GRADE")
    for r in result["grade"]["reasons"][:8]:
        L.append("   [%s] %s" % (r["grade"], r["reason"]))
    L.append("=" * width)
    return "\n".join(L)


def _wrap(s, width):
    words, line, out = (s or "").split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def html_report(result):
    """A one-page report, self-contained, meant to be read by someone who has to
    decide whether to sign for the hardware."""
    g = result["grade"]["grade"]
    color = GRADE_COLOR.get(g, "#555")
    e = html.escape

    def rows_probes():
        probes = result.get("probes") or {}
        order = {"DIVERGENT": 0, "ERROR": 0, "NAN": 0, "INF": 0, "SHAPE": 0,
                 "MISSING": 1, "WITHIN_TOLERANCE": 2, "IDENTICAL": 3}
        out = []
        for pid, rec in sorted(probes.items(),
                               key=lambda kv: (order.get(kv[1].get("verdict"), 9), kv[0])):
            v = rec.get("verdict", "?")
            cls = "bad" if v in ("DIVERGENT", "ERROR", "NAN", "INF", "SHAPE") else (
                "warn" if v in ("WITHIN_TOLERANCE", "MISSING") else "good")
            diff = ("%.2e" % rec["max_rel_diff"]) if rec.get("max_rel_diff") else "&mdash;"
            out.append(
                "<tr><td><code>%s</code></td><td>%s</td><td class='%s'>%s</td>"
                "<td class='num'>%s</td><td class='why'>%s</td></tr>" % (
                    e(pid), e(rec.get("title", "")), cls, v, diff,
                    e(rec.get("detail") or rec.get("symptom") or "")))
        return "\n".join(out)

    def rows_layers():
        out = []
        for r in (result.get("layers") or []):
            v = r.get("verdict")
            if v == "IDENTICAL":
                continue
            cls = "bad" if v not in ("WITHIN_TOLERANCE",) else "warn"
            diff = ("%.2e" % r["scaled_diff"]) if r.get("scaled_diff") is not None else "&mdash;"
            out.append("<tr><td><code>%s</code></td><td class='%s'>%s</td>"
                       "<td class='num'>%s</td></tr>" % (e(r.get("layer", "")), cls, v, diff))
        return "\n".join(out) or "<tr><td colspan=3>every layer is bit-identical</td></tr>"

    def blocks_why():
        out = []
        for a in (result.get("attribution") or [])[:8]:
            causes = "".join(
                "<li><b>%s</b><span>%s</span></li>" % (e(c["title"]), e(c["symptom"] or ""))
                for c in a["causes"]) or "<li>No probe explains this one.</li>"
            out.append("<div class='why-block'><h4><code>%s</code> diverges</h4><ul>%s</ul></div>"
                       % (e(a["layer"]), causes))
        return "\n".join(out)

    reasons = "".join("<li><span class='pill' style='background:%s'>%s</span>%s</li>"
                      % (GRADE_COLOR.get(r["grade"], "#555"), r["grade"], e(r["reason"]))
                      for r in result["grade"]["reasons"][:10])

    train = result.get("training") or {}
    train_html = ""
    if train.get("loss_gap") is not None:
        train_html = """
    <section><h2>Training</h2>
    <table class='kv'>
      <tr><th>steps</th><td>%d</td></tr>
      <tr><th>final loss, reference</th><td>%.6f</td></tr>
      <tr><th>final loss, target</th><td>%.6f</td></tr>
      <tr><th>separation</th><td class='%s'>%.3e</td></tr>
    </table></section>""" % (
            train.get("steps", 0), train["ref_final"], train["got_final"],
            "bad" if train["loss_gap"] > 1e-2 else "warn", train["loss_gap"])

    from string import Template
    tpl = Template(_HTML)
    return tpl.safe_substitute(
        color=color, grade=g,
        reference=e(result.get("reference", "")),
        target=e(result.get("target", "")),
        verdict=e(SUMMARY.get(g, "")),
        probes=rows_probes(), layers=rows_layers(), why=blocks_why(),
        training=train_html, reasons=reasons,
        version=e(result.get("version", "")), created=e(result.get("created", "")))



_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alignment Report</title>
<style>
:root{--bg:#faf8f4;--fg:#1a1a1a;--muted:#6b6b6b;--line:#e2ddd3;--card:#fff;}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15171a;--fg:#e9e6e1;--muted:#9a9a9a;--line:#2c2f34;--card:#1c1f23;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:40px 16px 80px;}
header{display:flex;gap:24px;align-items:center;border-bottom:2px solid var(--fg);padding-bottom:20px;margin-bottom:28px;flex-wrap:wrap}
.grade{width:92px;height:92px;border-radius:12px;display:grid;place-items:center;color:#fff;font-size:52px;font-weight:700;flex:none}
h1{font-size:21px;margin:0 0 6px;letter-spacing:.01em}
.sub{color:var(--muted);font-size:14px;margin:0}
.verdict{font-size:16px;margin:18px 0 0}
section{margin:34px 0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:8px;margin:0 0 14px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{text-align:left;font-weight:600;color:var(--muted);padding:6px 10px 6px 0;vertical-align:top}
td{padding:7px 10px 7px 0;border-top:1px solid var(--line);vertical-align:top}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--card);padding:1px 5px;border-radius:4px}
.good{color:#1d7a4c}.warn{color:#a86b1a}.bad{color:#b3401f;font-weight:600}
.num{font:12.5px ui-monospace,monospace;text-align:right;white-space:nowrap}
.why{color:var(--muted);font-size:12.5px}
.kv th{width:200px}
ul{margin:0;padding-left:0;list-style:none}
.reasons li{padding:7px 0;border-top:1px solid var(--line);display:flex;gap:10px;align-items:baseline}
.pill{color:#fff;border-radius:4px;padding:1px 8px;font-size:12px;font-weight:700;flex:none}
.why-block{margin-bottom:16px}
.why-block h4{margin:0 0 6px;font-size:14px;font-weight:600}
.why-block li{padding:6px 0 6px 14px;border-left:2px solid var(--line);margin-bottom:6px}
.why-block li span{display:block;color:var(--muted);font-size:12.5px}
footer{margin-top:50px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
@media(max-width:560px){.grade{width:68px;height:68px;font-size:38px}}
</style></head><body><div class="wrap">
<header>
  <div class="grade" style="background:$color">$grade</div>
  <div>
    <h1>Environment alignment report</h1>
    <p class="sub">reference &nbsp;$reference</p>
    <p class="sub">target &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$target</p>
  </div>
</header>
<p class="verdict">$verdict</p>
<section><h2>Assumptions</h2>
<table><tr><th>probe</th><th>assumption</th><th>verdict</th><th class="num">error</th><th>meaning</th></tr>
$probes</table></section>
<section><h2>Forward pass</h2>
<table><tr><th>layer</th><th>verdict</th><th class="num">error</th></tr>
$layers</table></section>
<section><h2>Why</h2>$why</section>
$training
<section><h2>What drove the grade</h2><ul class="reasons">$reasons</ul></section>
<footer><span>Obelisk $version</span><span>$created</span></footer>
</div></body></html>"""

def save_json(result, path):
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1)
    return path
