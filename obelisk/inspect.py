"""Static inspection: what does this repository assume, before anything runs.

This is the part that needs no accelerator and no access to the code's data.  It
reads the source, finds the places where the program has written a hardware
belief into itself, and says which of those beliefs the target device is known
not to hold.
"""

import os
import re

# Each rule: id, regex, which assumption it belongs to, severity, what it means.
RULES = [
    ("api.cuda_call", r"\.cuda\s*\(", "api", "high",
     "a tensor or module is moved to CUDA by name"),
    ("api.device_string", r"""["']cuda(:\d+)?["']""", "api", "high",
     "a device is named as a string literal"),
    ("api.torch_cuda", r"torch\.cuda\.", "api", "high",
     "the torch.cuda namespace is used directly"),
    ("api.amp_cuda", r"""torch\.cuda\.amp|autocast\(\s*["']cuda""", "api", "medium",
     "automatic mixed precision is pinned to CUDA"),
    ("api.nccl", r"nccl", "api", "high",
     "the NCCL collective library is requested by name"),
    ("kernel.triton", r"import\s+triton|@triton\.jit", "kernel", "critical",
     "hand-written Triton kernels, which are compiled for a specific target"),
    ("kernel.flash_attn", r"flash_attn|flash-attn|FlashAttention", "kernel", "critical",
     "FlashAttention, whose kernels are written against specific hardware"),
    ("kernel.cuda_extension", r"CUDAExtension|load\s*\(\s*name\s*=|\.cu['\"]",
     "kernel", "critical", "a compiled CUDA extension is built from source"),
    ("kernel.xformers", r"xformers|apex|bitsandbytes|deepspeed", "kernel", "high",
     "a third-party kernel library with hardware-specific code paths"),
    ("kernel.warp", r"warpSize|__shfl|blockDim\.x\s*==\s*32|threadIdx", "kernel", "critical",
     "CUDA source that assumes a particular warp size or thread layout"),
    ("precision.tf32", r"allow_tf32", "precision", "high",
     "reduced-precision float32 is switched on or off explicitly"),
    ("precision.half", r"\.half\s*\(|float16|fp16", "precision", "medium",
     "half precision, whose accumulator width is device-defined"),
    ("precision.bf16", r"bfloat16|bf16", "precision", "medium",
     "bfloat16, which is not available on every accelerator"),
    ("order.atomics", r"index_add_|scatter_add_|index_put_|\.scatter_\(",
     "ordering", "high",
     "an accumulation whose order depends on how threads are scheduled"),
    ("order.deterministic", r"use_deterministic_algorithms|cudnn\.deterministic",
     "ordering", "low", "determinism is being requested explicitly"),
    ("rng.seed", r"manual_seed|torch\.seed\(|np\.random\.seed", "randomness", "medium",
     "a seed is set, which only reproduces results on one device family"),
    ("rng.dropout", r"[Dd]ropout", "randomness", "medium",
     "dropout, which consumes the device random stream"),
    ("mem.pinned", r"pin_memory|non_blocking\s*=\s*True", "memory", "low",
     "pinned or asynchronous transfers"),
    ("mem.channels_last", r"channels_last", "memory", "medium",
     "an explicit memory format, which not every backend honours"),
    ("build.cuda_home", r"CUDA_HOME|CUDA_VISIBLE_DEVICES|nvcc", "build", "medium",
     "the build or launch is wired to the CUDA toolchain"),
]

COMPILED = [(rid, re.compile(rx), group, sev, why) for rid, rx, group, sev, why in RULES]

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build",
             "dist", ".mypy_cache", ".pytest_cache", "site-packages", ".idea"}
CODE_EXT = {".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".sh", ".yaml",
            ".yml", ".toml", ".cfg", ".txt", ".md"}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def scan(root, max_files=4000):
    """Walk a tree and record every assumption the source makes."""
    root = os.path.abspath(os.path.expanduser(root))
    findings = []
    files_scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in CODE_EXT:
                continue
            path = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(path) > 2 * 1024 * 1024:
                    continue
                with open(path, "r", errors="ignore") as fh:
                    lines = fh.readlines()
            except Exception:
                continue
            files_scanned += 1
            if files_scanned > max_files:
                break
            rel = os.path.relpath(path, root)
            for lineno, line in enumerate(lines, 1):
                if len(line) > 1000:
                    continue
                for rid, rx, group, sev, why in COMPILED:
                    if rx.search(line):
                        findings.append({
                            "rule": rid,
                            "group": group,
                            "severity": sev,
                            "meaning": why,
                            "file": rel,
                            "line": lineno,
                            "text": line.strip()[:200],
                        })
    return {"root": root, "files_scanned": files_scanned, "findings": findings}


def summarise(result):
    """Group the raw hits into one row per rule."""
    by_rule = {}
    for f in result["findings"]:
        r = by_rule.setdefault(f["rule"], {
            "rule": f["rule"], "group": f["group"], "severity": f["severity"],
            "meaning": f["meaning"], "count": 0, "examples": [],
        })
        r["count"] += 1
        if len(r["examples"]) < 3:
            r["examples"].append("%s:%d" % (f["file"], f["line"]))
    rows = sorted(by_rule.values(),
                  key=lambda r: (SEVERITY_ORDER.get(r["severity"], 9), -r["count"]))
    return rows


def predicted_risk(rows):
    """A grade from the source alone, before any device has been touched.

    This is a prediction, and it is written down before the run so that it can be
    held against the result afterwards.
    """
    crit = [r for r in rows if r["severity"] == "critical"]
    high = [r for r in rows if r["severity"] == "high"]
    if crit:
        grade = "D"
        why = ("%d assumption%s in this repository cannot hold on a different "
               "accelerator without being rewritten" % (len(crit), "" if len(crit) == 1 else "s"))
    elif len(high) >= 3:
        grade = "C"
        why = "%d high-severity assumptions, none of them fatal on their own" % len(high)
    elif high:
        grade = "B"
        why = "a small number of high-severity assumptions"
    elif rows:
        grade = "B"
        why = "only low-severity assumptions"
    else:
        grade = "A"
        why = "no hardware assumptions found in the source"
    return {
        "grade": grade,
        "why": why,
        "critical": [r["rule"] for r in crit],
        "high": [r["rule"] for r in high],
    }
