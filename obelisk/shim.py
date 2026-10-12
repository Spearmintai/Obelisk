"""Making the program run, and saying exactly what that cost.

A missing operator is usually not a missing capability.  log1p is log(1+x),
asinh is a logarithm and a square root, a standard deviation is the square root
of a variance.  The hardware can nearly always do the work; what it lacks is
someone to write the arrangement down.

So the shim fills those in.  Every substitution is one of three kinds, and every
one of them is recorded:

  rewrite   expressed in operators the device does have, still on the device
  fallback  computed on the CPU, which is correct and slow, and the ledger says
            how slow
  correct   the device has the operator, and the operator is wrong, so it is
            replaced with an implementation that is not

The point is not that the program stops crashing.  Anything can stop a program
crashing.  The point is the ledger: at the end of the run you know precisely
which parts of your computation did not happen the way you wrote them.
"""

import math
import time

LN2 = 0.6931471805599453

# Named here as well as implemented below, so that a machine with no PyTorch can
# still say what the shim would be able to do.
REWRITE_NAMES = (
    "log1p expm1 exp2 sinh cosh asin arcsin asinh arcsinh acosh arccosh atanh "
    "arctanh erfc logit sinc hypot copysign signbit trunc fix logaddexp "
    "logaddexp2 logical_xor isposinf isneginf heaviside clamp_min clamp_max "
    "float_power rsub nansum nanmean std std_mean median kthvalue "
    "F.logsigmoid F.mish F.softshrink F.hardshrink").split()

CORRECTION_NAMES = ("selu", "F.selu")

FALLBACK_NAMES = (
    "lgamma digamma mvlgamma i0 erfinv polygamma det logdet slogdet svd qr "
    "cholesky matrix_exp triangular_solve cholesky_solve lu cdist pdist "
    "nanmedian cummax cummin aminmax angle frexp gcd lcm nextafter "
    "logcumsumexp").split()


def handled():
    """Every operator the shim can stand in for, and how."""
    out = {}
    for n in REWRITE_NAMES:
        out[n] = "rewrite"
    for n in CORRECTION_NAMES:
        out[n] = "correct"
    for n in FALLBACK_NAMES:
        out[n] = "fallback"
    return out


class Ledger:
    """What the shim did, and what it cost."""

    def __init__(self):
        self.entries = {}
        self.decisions = {}

    def decide(self, name, strategy, reason):
        self.decisions[name] = {"strategy": strategy, "reason": reason}

    def record(self, name, seconds):
        e = self.entries.setdefault(name, {"calls": 0, "seconds": 0.0})
        e["calls"] += 1
        e["seconds"] += seconds

    def rows(self):
        out = []
        for name, d in sorted(self.decisions.items()):
            used = self.entries.get(name, {"calls": 0, "seconds": 0.0})
            out.append({
                "op": name,
                "strategy": d["strategy"],
                "reason": d["reason"],
                "calls": used["calls"],
                "seconds": used["seconds"],
            })
        return out

    def summary(self):
        rows = self.rows()
        used = [r for r in rows if r["calls"]]
        return {
            "patched": len(rows),
            "exercised": len(used),
            "calls": sum(r["calls"] for r in used),
            "seconds": sum(r["seconds"] for r in used),
            "by_strategy": {
                s: len([r for r in rows if r["strategy"] == s])
                for s in ("rewrite", "fallback", "correct")
            },
            "fallback_seconds": sum(r["seconds"] for r in used
                                    if r["strategy"] == "fallback"),
        }


# --------------------------------------------------------------- rewrites
# Each one is the operator written out in terms of operators that exist.

def _rewrites(torch):
    F = torch.nn.functional

    def std(x, *a, **k):
        unbiased = k.pop("unbiased", True)
        if a and isinstance(a[0], bool):
            unbiased, a = a[0], a[1:]
        return torch.sqrt(torch.var(x, *a, unbiased=unbiased, **k))

    def std_mean(x, *a, **k):
        return std(x, *a, **k), torch.mean(x, *a, **k)

    def median(x, dim=None, keepdim=False):
        if dim is None:
            flat = x.reshape(-1).sort().values
            return flat[(flat.numel() - 1) // 2]
        values = x.sort(dim=dim).values
        index = (x.shape[dim] - 1) // 2
        picked = values.index_select(dim, torch.tensor([index], device=x.device))
        if not keepdim:
            picked = picked.squeeze(dim)
        return picked

    def kthvalue(x, k, dim=-1, keepdim=False):
        s = x.sort(dim=dim)
        idx = torch.tensor([k - 1], device=x.device)
        v = s.values.index_select(dim, idx)
        i = s.indices.index_select(dim, idx)
        if not keepdim:
            v, i = v.squeeze(dim), i.squeeze(dim)
        return torch.return_types.kthvalue((v, i)) if hasattr(
            torch.return_types, "kthvalue") else (v, i)

    return {
        "log1p": lambda x: torch.log(1.0 + x),
        "expm1": lambda x: torch.exp(x) - 1.0,
        "exp2": lambda x: torch.exp(x * LN2),
        "sinh": lambda x: (torch.exp(x) - torch.exp(-x)) * 0.5,
        "cosh": lambda x: (torch.exp(x) + torch.exp(-x)) * 0.5,
        "asin": lambda x: torch.atan2(x, torch.sqrt((1.0 - x * x).clamp(min=0))),
        "arcsin": lambda x: torch.atan2(x, torch.sqrt((1.0 - x * x).clamp(min=0))),
        "asinh": lambda x: torch.log(x + torch.sqrt(x * x + 1.0)),
        "arcsinh": lambda x: torch.log(x + torch.sqrt(x * x + 1.0)),
        "acosh": lambda x: torch.log(x + torch.sqrt((x * x - 1.0).clamp(min=0))),
        "arccosh": lambda x: torch.log(x + torch.sqrt((x * x - 1.0).clamp(min=0))),
        "atanh": lambda x: 0.5 * torch.log((1.0 + x) / (1.0 - x)),
        "arctanh": lambda x: 0.5 * torch.log((1.0 + x) / (1.0 - x)),
        "erfc": lambda x: 1.0 - torch.erf(x),
        "logit": lambda x, eps=None: torch.log(
            (x.clamp(eps, 1 - eps) if eps is not None else x) /
            (1.0 - (x.clamp(eps, 1 - eps) if eps is not None else x))),
        "sinc": lambda x: torch.where(x == 0, torch.ones_like(x),
                                      torch.sin(math.pi * x) / (math.pi * x)),
        "hypot": lambda a, b: torch.sqrt(a * a + b * b),
        "copysign": lambda a, b: torch.abs(a) * torch.where(
            b < 0, -torch.ones_like(b), torch.ones_like(b)),
        "signbit": lambda x: x < 0,
        "trunc": lambda x: torch.where(x >= 0, torch.floor(x), torch.ceil(x)),
        "fix": lambda x: torch.where(x >= 0, torch.floor(x), torch.ceil(x)),
        "logaddexp": lambda a, b: torch.maximum(a, b) + torch.log(
            1.0 + torch.exp(-torch.abs(a - b))),
        "logaddexp2": lambda a, b: torch.maximum(a, b) + torch.log(
            1.0 + torch.exp(-torch.abs(a - b) * LN2)) / LN2,
        "logical_xor": lambda a, b: (a != 0) != (b != 0),
        "isposinf": lambda x: x == float("inf"),
        "isneginf": lambda x: x == float("-inf"),
        "heaviside": lambda x, v: torch.where(
            x < 0, torch.zeros_like(x), torch.where(x > 0, torch.ones_like(x), v)),
        "clamp_min": lambda x, v: torch.clamp(x, min=v),
        "clamp_max": lambda x, v: torch.clamp(x, max=v),
        "float_power": lambda x, y: torch.pow(x, y),
        "rsub": lambda x, y, alpha=1: y - alpha * x,
        "nansum": lambda x, *a, **k: torch.sum(
            torch.where(torch.isnan(x), torch.zeros_like(x), x), *a, **k),
        "nanmean": lambda x, *a, **k: (
            torch.sum(torch.where(torch.isnan(x), torch.zeros_like(x), x), *a, **k) /
            torch.sum((~torch.isnan(x)).to(x.dtype), *a, **k)),
        "std": std,
        "std_mean": std_mean,
        "median": median,
        "kthvalue": kthvalue,
        "F.logsigmoid": lambda x: -torch.log(1.0 + torch.exp(-torch.abs(x))) + \
                                  torch.clamp(x, max=0),
        "F.mish": lambda x, inplace=False: x * torch.tanh(
            torch.log(1.0 + torch.exp(-torch.abs(x))) + torch.clamp(x, min=0)),
        "F.softshrink": lambda x, lambd=0.5: torch.where(
            x > lambd, x - lambd, torch.where(x < -lambd, x + lambd, torch.zeros_like(x))),
        "F.hardshrink": lambda x, lambd=0.5: torch.where(
            torch.abs(x) > lambd, x, torch.zeros_like(x)),
    }


def _corrections(torch):
    """Operators the device provides and gets wrong."""
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717

    def selu(x, inplace=False):
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1.0))

    return {"selu": selu, "F.selu": selu}


# ------------------------------------------------------- correctness checks
# A missing operator announces itself.  A wrong one does not, so each of these
# is checked against its own definition before the program is trusted with it.

def _checks(torch):
    def selu_check(fn, dev):
        x = torch.tensor([-2.0, -0.5, 0.5, 3.061628], device=dev)
        scale = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        exact = scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1.0))
        got = fn(x)
        return float((got - exact).abs().max())

    return {"selu": selu_check, "F.selu": selu_check}


MATRIX_OPS = {"det", "logdet", "slogdet", "svd", "qr", "geqrf", "lu",
              "matrix_exp", "nuclear_norm", "pca_lowrank", "svd_lowrank"}
SPD_OPS = {"cholesky"}
PAIR_MATRIX_OPS = {"triangular_solve", "cholesky_solve", "cdist"}


def _probe_inputs(torch, name, dev):
    """A set of arguments this operator will accept, so that the probe measures
    the device rather than the caller."""
    unary_unit = torch.tensor([0.1, 0.3, 0.6], device=dev)
    unary = torch.tensor([-1.5, 0.4, 2.0], device=dev)
    positive = torch.tensor([1.5, 2.0, 3.0], device=dev)

    if name in MATRIX_OPS or name in SPD_OPS or name in PAIR_MATRIX_OPS:
        g = torch.Generator().manual_seed(11)
        m = torch.randn(4, 4, generator=g)
        spd = (m @ m.t() + 4 * torch.eye(4)).to(dev)
        m = m.to(dev)
        if name in SPD_OPS:
            return (spd,)
        if name in PAIR_MATRIX_OPS:
            return (spd, spd)
        return (m,)
    if name in ("pdist",):
        g = torch.Generator().manual_seed(12)
        return (torch.randn(5, 3, generator=g).to(dev),)
    if name in ("asin", "arcsin", "atanh", "arctanh", "logit"):
        return (unary_unit,)
    if name in ("acosh", "arccosh"):
        return (positive,)
    if name in ("hypot", "copysign", "logaddexp", "logaddexp2", "logical_xor",
                "heaviside", "float_power", "rsub", "nextafter", "gcd", "lcm"):
        if name in ("gcd", "lcm"):
            a = torch.tensor([6, 8, 12], device=dev)
            return (a, a + 2)
        return (unary, positive)
    if name in ("clamp_min", "clamp_max"):
        return (unary, 0.0)
    if name == "kthvalue":
        return (unary, 2)
    if name == "mvlgamma":
        return (positive, 2)
    if name == "polygamma":
        return (1, positive)
    return (unary,)


def _resolve(torch, name):
    """Find the live function behind a dotted name, and where to put it back."""
    if name.startswith("F."):
        return torch.nn.functional, name[2:]
    return torch, name


def activate(backend=None, verbose=False):
    """Patch what this device cannot do, or does wrongly, and start a ledger."""
    import torch

    from . import backend as B
    be = backend or B.pick("auto")
    dev = be.device

    ledger = Ledger()
    rewrites = _rewrites(torch)
    corrections = _corrections(torch)
    checks = _checks(torch)

    def install(name, impl, strategy, reason):
        holder, attr = _resolve(torch, name)
        ledger.decide(name, strategy, reason)

        def patched(*args, **kwargs):
            start = time.time()
            try:
                return impl(*args, **kwargs)
            finally:
                ledger.record(name, time.time() - start)

        patched.__name__ = attr
        patched.__doc__ = "Obelisk %s: %s" % (strategy, reason)
        setattr(holder, attr, patched)
        if holder is torch and hasattr(torch.Tensor, attr):
            try:
                setattr(torch.Tensor, attr,
                        lambda self, *a, **k: patched(self, *a, **k))
            except (TypeError, AttributeError):
                pass
        if verbose:
            print("    %-18s %-9s %s" % (name, strategy, reason))

    def cpu_fallback(original, name):
        def run(*args, **kwargs):
            moved = tuple(a.detach().to("cpu") if hasattr(a, "device") else a
                          for a in args)
            kw = {k: (v.detach().to("cpu") if hasattr(v, "device") else v)
                  for k, v in kwargs.items()}
            out = original(*moved, **kw)
            back = args[0].device if args and hasattr(args[0], "device") else dev
            if hasattr(out, "to"):
                return out.to(back)
            if isinstance(out, tuple):
                return tuple(o.to(back) if hasattr(o, "to") else o for o in out)
            return out
        return run

    # 1. Operators that are simply absent.
    for name, impl in rewrites.items():
        holder, attr = _resolve(torch, name)
        fn = getattr(holder, attr, None)
        if fn is None:
            install(name, impl, "rewrite", "not present in this build")
            continue
        try:
            fn(*_probe_inputs(torch, attr, dev))
            be.synchronize()
        except Exception as exc:
            text = str(exc).split("\n")[0]
            missing = ("not implement" in text.lower() or
                       "could not run" in text.lower())
            try:
                impl(*_probe_inputs(torch, attr, dev))
                be.synchronize()
            except Exception as sub_exc:
                if verbose:
                    print("    %-18s left alone (%s)" % (name, str(sub_exc)[:44]))
                continue
            install(name, impl, "rewrite",
                    "no kernel on this device" if missing
                    else "device kernel fails: %s" % text[:60])

    # 2. Operators that are present and wrong.
    for name, check in checks.items():
        holder, attr = _resolve(torch, name)
        fn = getattr(holder, attr, None)
        if fn is None or name in ledger.decisions:
            continue
        try:
            error = check(fn, dev)
        except Exception:
            continue
        if error > 1e-4:
            install(name, corrections[name], "correct",
                    "device kernel is wrong by %.2e" % error)

    # 3. Operators with no rewrite, which have to go back to the CPU.
    for name in FALLBACK_NAMES:
        holder, attr = _resolve(torch, name)
        fn = getattr(holder, attr, None)
        if fn is None or name in ledger.decisions:
            continue
        try:
            fn(*_probe_inputs(torch, attr, dev))
            be.synchronize()
        except Exception as exc:
            text = str(exc).split("\n")[0]
            missing = ("not implement" in text.lower() or
                       "could not run" in text.lower())
            install(name, cpu_fallback(fn, name), "fallback",
                    ("no kernel on this device; computed on the CPU" if missing
                     else "device kernel fails (%s); computed on the CPU"
                          % text[:50]))

    return ledger
