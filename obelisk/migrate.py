"""Rewriting a repository from one accelerator to another.

The rewrite itself is mechanical and small.  It is included not because it is
hard but because it is the step everyone does by hand, and because doing it
mechanically is what makes the before and after comparable: the only thing that
changed is listed in the patch.
"""

import difflib
import os
import re

TARGETS = {
    "musa": {
        "module": "torch_musa",
        "namespace": "torch.musa",
        "device": "musa",
        "method": "musa",
    },
    "cuda": {
        "module": None,
        "namespace": "torch.cuda",
        "device": "cuda",
        "method": "cuda",
    },
}


def rewrite_text(text, target="musa"):
    """Return the rewritten source and a list of what was changed."""
    spec = TARGETS[target]
    changes = []

    def note(kind, n):
        if n:
            changes.append({"change": kind, "count": n})

    out = text

    if spec["module"] and re.search(r"^\s*import\s+torch\s*$", out, re.M):
        if spec["module"] not in out:
            out, n = re.subn(r"(^\s*import\s+torch\s*$)",
                             r"\1\nimport %s  # obelisk: target backend" % spec["module"],
                             out, count=1, flags=re.M)
            note("import %s" % spec["module"], n)

    out, n = re.subn(r"\.cuda\s*\(", ".%s(" % spec["method"], out)
    note(".cuda() -> .%s()" % spec["method"], n)

    out, n = re.subn(r"""(["'])cuda(:\d+)?\1""",
                     lambda m: "%s%s%s%s" % (m.group(1), spec["device"],
                                             m.group(2) or "", m.group(1)), out)
    note("device string -> %r" % spec["device"], n)

    # Device names also appear inside format strings, where the index is an
    # expression rather than a literal digit.
    out, n = re.subn(r"cuda:\{", spec["device"] + ":{", out)
    note("device name inside a format string", n)

    out, n = re.subn(r"torch\.cuda\.", spec["namespace"] + ".", out)
    note("torch.cuda.* -> %s.*" % spec["namespace"], n)

    out, n = re.subn(r"torch\.backends\.cuda\.", "torch.backends.%s." % spec["device"], out)
    note("torch.backends.cuda.* -> torch.backends.%s.*" % spec["device"], n)

    return out, changes


UNPORTABLE = [
    (re.compile(r"import\s+triton|@triton\.jit"), "Triton kernels"),
    (re.compile(r"flash_attn|FlashAttention"), "FlashAttention"),
    (re.compile(r"CUDAExtension|\.cu['\"]"), "compiled CUDA extensions"),
    (re.compile(r"xformers|apex|bitsandbytes|deepspeed"), "third-party kernel libraries"),
]


def plan(root, target="musa", apply=False):
    """Rewrite every Python file under root, returning a patch and a list of the
    places a mechanical rewrite is not enough."""
    root = os.path.abspath(os.path.expanduser(root))
    diffs = []
    changed_files = 0
    all_changes = {}
    blockers = []
    leftovers = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            try:
                with open(path, errors="ignore") as fh:
                    src = fh.read()
            except Exception:
                continue

            for rx, label in UNPORTABLE:
                if rx.search(src):
                    blockers.append({"file": rel, "needs": label})

            new, changes = rewrite_text(src, target)

            # Whatever still names the old backend after the rewrite is residue:
            # the rewrite could not prove it safe to touch, so it is reported
            # rather than silently left behind.
            if target != "cuda":
                residue = [(i + 1, ln.strip()[:120])
                           for i, ln in enumerate(new.splitlines())
                           if re.search(r"cuda", ln, re.I)
                           and not ln.strip().startswith("#")]
                for lineno, textline in residue[:40]:
                    leftovers.append({"file": rel, "line": lineno, "text": textline})

            if new != src:
                changed_files += 1
                for c in changes:
                    all_changes[c["change"]] = all_changes.get(c["change"], 0) + c["count"]
                diff = difflib.unified_diff(
                    src.splitlines(True), new.splitlines(True),
                    fromfile="a/" + rel, tofile="b/" + rel)
                diffs.append("".join(diff))
                if apply:
                    with open(path, "w") as fh:
                        fh.write(new)

    seen = set()
    unique_blockers = []
    for b in blockers:
        key = (b["file"], b["needs"])
        if key not in seen:
            seen.add(key)
            unique_blockers.append(b)

    return {
        "target": target,
        "leftovers": leftovers,
        "files_changed": changed_files,
        "changes": [{"change": k, "count": v} for k, v in sorted(all_changes.items())],
        "blockers": unique_blockers,
        "patch": "".join(diffs),
        "applied": bool(apply),
    }
