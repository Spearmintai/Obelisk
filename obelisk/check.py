"""Judging a repository against a card that is somewhere else.

This is the arrangement the work actually takes.  The code lives on a laptop or a
build server.  The accelerator lives in a machine room, often with no route to
the internet, and usually nobody has put the code on it yet.  The two are never
in the same place at the same time, which is exactly why nobody checks.

Here the card is measured once, on its own, with none of your code present.  The
profile it produces travels.  The question "will this repository run on that
card" is then answered where the repository already is.
"""

import ast
import os

from .coverage import is_stochastic

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build",
             "dist", ".mypy_cache", ".pytest_cache", "site-packages"}

# Modules whose attribute calls are torch operators under another name.
TORCH_ALIASES = {"torch", "th"}
FUNCTIONAL_ALIASES = {"F", "functional"}

# Modules that share names with torch operators and have nothing to do with it.
# tl is Triton's language namespace, whose exp2 runs on the GPU but is compiled
# by Triton rather than dispatched by PyTorch.
NOT_TORCH = {
    "np", "numpy", "math", "cmath", "scipy", "sp", "pd", "pandas", "random",
    "os", "sys", "re", "json", "time", "datetime", "itertools", "statistics",
    "tl", "triton", "tlx", "cp", "cupy", "jnp", "jax", "tf", "plt", "nn",
    "self", "cls", "args", "config", "cfg", "parser", "logger", "logging",
}


class _Visitor(ast.NodeVisitor):
    def __init__(self, path, imports_torch=True):
        self.path = path
        self.calls = []
        self.aliases = dict()
        # A bare method call such as x.std() is only evidence of a torch call if
        # this file has any torch in it at all.
        self.imports_torch = imports_torch

    def visit_Import(self, node):
        for a in node.names:
            if a.name == "torch":
                self.aliases[a.asname or "torch"] = "torch"
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("torch"):
            for a in node.names:
                if a.name in ("functional",) or node.module.endswith("functional"):
                    self.aliases[a.asname or a.name] = "F"
        self.generic_visit(node)

    def _dotted(self, node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return list(reversed(parts))
        return None

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute):
            dotted = self._dotted(func)
            if dotted:
                self._record(dotted, node.lineno)
            else:
                self.calls.append(("method", func.attr, node.lineno))
        self.generic_visit(node)

    def _record(self, parts, lineno):
        name = parts[-1]
        root = parts[0]
        if root in TORCH_ALIASES or self.aliases.get(root) == "torch":
            if "functional" in parts:
                self.calls.append(("F", name, lineno))
            else:
                self.calls.append(("torch", name, lineno))
        elif root in FUNCTIONAL_ALIASES or self.aliases.get(root) == "F":
            self.calls.append(("F", name, lineno))
        elif root in NOT_TORCH or len(parts) > 2:
            # Another library's namespace, or a path too deep to be a tensor
            # method; neither is evidence about this device.
            return
        elif self.imports_torch:
            # Possibly a tensor method.  Recorded, but marked as weaker.
            self.calls.append(("method", name, lineno))


def scan_calls(root):
    """Every operator name this repository calls, and where."""
    root = os.path.abspath(os.path.expanduser(root))
    uses = {}
    files = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, errors="ignore") as fh:
                    source = fh.read()
                tree = ast.parse(source, path)
            except Exception:
                continue
            files += 1
            rel = os.path.relpath(path, root)
            v = _Visitor(rel, imports_torch=("import torch" in source))
            v.visit(tree)
            for kind, name, lineno in v.calls:
                uses.setdefault((kind, name), []).append((rel, lineno))
    return {"root": root, "files": files, "uses": uses}


def check(repo, profile, shim_table=None):
    """Which of this repository's calls will not work on that card."""
    cov = (profile.get("payload") or {}).get("coverage") or {}
    status = {}
    for rec in cov.get("operators", []):
        status[rec["op"]] = rec

    dtypes = (profile.get("payload") or {}).get("dtypes") or {}
    scan = scan_calls(repo)
    shim_table = shim_table or {}

    merged = {}
    confirmed_ok = 0

    for (kind, name), sites in sorted(scan["uses"].items()):
        if kind == "F":
            candidates = ["F." + name, "torch." + name]
        else:
            candidates = ["torch." + name]

        rec = key = None
        for c in candidates:
            if c in status:
                rec, key = status[c], c
                break
        if rec is None or is_stochastic(key) or rec.get("status") == "STOCHASTIC":
            continue

        if rec["status"] in ("EXACT", "CLOSE") and \
                not (rec["status"] == "CLOSE" and rec.get("error", 0) > 1e-5):
            confirmed_ok += 1
            continue

        shim = shim_table.get(key) or shim_table.get(name) or \
            shim_table.get("F." + name)
        row = merged.setdefault(key, {
            "op": key, "status": rec["status"], "sites": [], "count": 0,
            "shim": shim, "detail": rec.get("detail"), "error": rec.get("error"),
            "weak": True,
        })
        row["sites"].extend(sites)
        row["count"] += len(sites)
        if kind != "method":
            row["weak"] = False

    blocking, wrong, risky = [], [], []
    for row in merged.values():
        row["sites"] = sorted(set(row["sites"]))[:8]
        if row["status"] in ("MISSING", "ERROR"):
            blocking.append(row)
        elif row["status"] in ("MISMATCH", "SHAPE"):
            wrong.append(row)
        else:
            risky.append(row)

    dtype_notes = []
    for name, row in sorted(dtypes.items()):
        integral = name.startswith(("int", "uint", "bool"))
        broken = []
        for k, v in row.items():
            if v is not False or k == "detail":
                continue
            if integral and k == "matmul":
                continue
            broken.append(k)
        if row.get("allocate") is False:
            broken = ["allocate"]
        if broken:
            dtype_notes.append({"dtype": name, "unsupported": broken})

    return {
        "repo": scan["root"],
        "files": scan["files"],
        "card": (profile.get("device") or {}).get("name"),
        "card_backend": (profile.get("device") or {}).get("backend"),
        "card_digest": profile.get("digest"),
        "blocking": sorted(blocking, key=lambda r: -r["count"]),
        "wrong": sorted(wrong, key=lambda r: -r["count"]),
        "risky": sorted(risky, key=lambda r: -r["count"]),
        "confirmed_ok": confirmed_ok,
        "dtypes": dtype_notes,
    }


def verdict(result):
    """A grade, and the sentence that justifies it.

    A call written as torch.std is certain.  One written as x.std() is probable,
    because x might not be a tensor.  Both count; the wording says which is
    which, and the grade never claims more than the findings below it.
    """
    blocking, wrong = result["blocking"], result["wrong"]
    hard = [r for r in blocking if not r["weak"]]
    soft = [r for r in blocking if r["weak"]]
    wrong_hard = [r for r in wrong if not r["weak"]]
    wrong_soft = [r for r in wrong if r["weak"]]
    unshimmable = [r for r in hard if not r["shim"]]

    def count(n, noun="operation"):
        return "%d %s%s" % (n, noun, "" if n == 1 else "s")

    if wrong_hard:
        return "D", ("%s this code calls returns a wrong answer on this card, "
                     "without raising" % count(len(wrong_hard)))
    if unshimmable:
        return "D", ("%s this code calls cannot run on this card and cannot be "
                     "substituted" % count(len(unshimmable)))
    if hard or soft or wrong_soft:
        parts = []
        if hard:
            parts.append("%s certainly" % count(len(hard)))
        if soft:
            parts.append("%s probably" % count(len(soft)))
        if wrong_soft:
            parts.append("%s possibly returning a wrong answer" % count(len(wrong_soft)))
        return "C", (" and ".join(parts) +
                     " would fail here, and every one can be substituted")
    if result["risky"]:
        return "B", "every operation runs; some return slightly different numbers"
    return "A", "every operation this code calls behaves the same on this card"
