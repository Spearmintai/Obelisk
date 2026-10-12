"""Device discovery across CUDA, MUSA (Moore Threads) and CPU."""

import os
import platform
import subprocess


class Backend:
    def __init__(self, kind, index=0):
        self.kind = kind
        self.index = index

    @property
    def device(self):
        if self.kind == "cpu":
            return "cpu"
        return "%s:%d" % (self.kind, self.index)

    def module(self):
        import torch
        if self.kind == "cuda":
            return torch.cuda
        if self.kind == "musa":
            import torch_musa  # noqa: F401
            return torch.musa
        return None

    def synchronize(self):
        m = self.module()
        if m is not None:
            try:
                m.synchronize()
            except Exception:
                pass

    def info(self):
        import torch
        out = {
            "backend": self.kind,
            "device": self.device,
            "torch": torch.__version__,
            "host": platform.node(),
            "arch": platform.machine(),
            "python": platform.python_version(),
        }
        m = self.module()
        if m is None:
            out["name"] = "cpu (%s)" % platform.processor() or "cpu"
            return out
        try:
            props = m.get_device_properties(self.index)
            out["name"] = getattr(props, "name", "?")
            total = getattr(props, "total_memory", None)
            if total:
                out["memory_gb"] = round(total / (1024 ** 3), 1)
            cap = getattr(props, "major", None)
            if cap is not None:
                out["capability"] = "%s.%s" % (props.major, getattr(props, "minor", 0))
            mp = getattr(props, "multi_processor_count", None)
            if mp:
                out["multiprocessors"] = mp
        except Exception as exc:
            out["name"] = "unknown"
            out["probe_error"] = str(exc)
        if self.kind == "cuda":
            out["runtime_version"] = torch.version.cuda
            out["driver"] = _cmd(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"])
            out["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
            try:
                out["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
            except Exception:
                pass
        if self.kind == "musa":
            import torch_musa
            out["runtime_version"] = getattr(torch_musa, "__version__", "?")
            out["driver"] = _cmd(["mthreads-gmi", "-q", "-i", "0"], grep="Driver Version")
            for flag in ("allow_tf32",):
                try:
                    out["allow_tf32_matmul"] = bool(getattr(torch.backends.musa.matmul, flag))
                except Exception:
                    pass
        return out


def _cmd(argv, grep=None):
    try:
        out = subprocess.check_output(argv, stderr=subprocess.DEVNULL).decode("utf8", "ignore")
    except Exception:
        return None
    if grep:
        for line in out.splitlines():
            if grep in line:
                return line.split(":")[-1].strip()
        return None
    return out.strip().splitlines()[0] if out.strip() else None


def available():
    """Return every backend usable in this process, best first."""
    found = []
    try:
        import torch
    except ImportError:
        return found
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                found.append(Backend("cuda", i))
    except Exception:
        pass
    try:
        import torch_musa  # noqa: F401
        if torch.musa.is_available():
            for i in range(torch.musa.device_count()):
                found.append(Backend("musa", i))
    except Exception:
        pass
    found.append(Backend("cpu"))
    return found


def pick(name=None):
    """Pick a backend by name ('cuda', 'musa', 'cpu', 'musa:1') or take the best."""
    found = available()
    if not found:
        raise RuntimeError("torch is not installed in this interpreter")
    if not name or name == "auto":
        return found[0]
    kind, _, idx = name.partition(":")
    idx = int(idx) if idx else 0
    for b in found:
        if b.kind == kind and b.index == idx:
            return b
    raise RuntimeError("backend %r not available here (have: %s)"
                       % (name, ", ".join(sorted({b.kind for b in found}))))
