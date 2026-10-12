"""The probe suite.

Each probe isolates exactly one assumption a program makes about its
accelerator, and turns it into tensors whose values differ if and only if the
assumption is broken.  The inputs are generated on the CPU from a fixed seed and
then moved to the device, so two devices always see identical bytes going in.
Whatever differs on the way out came from the device.

A probe is deliberately small.  It is not a benchmark and it is not a workload:
it is a question with a numerical answer.
"""

import math

REGISTRY = []


def probe(pid, title, layer, symptom, severity="high", expect="numeric",
          cross_device=True):
    """Register a probe.

    expect="bitwise"   the two devices have no excuse to differ at all
    cross_device=False the probe only means something on one device at a time
    """
    def wrap(fn):
        REGISTRY.append({
            "id": pid,
            "title": title,
            "layer": layer,
            "symptom": symptom,
            "severity": severity,
            "expect": expect,
            "cross_device": cross_device,
            "fn": fn,
        })
        return fn
    return wrap


def _gen(shape, dtype="float32", seed=0, scale=1.0, kind="normal"):
    """Deterministic CPU-side input.  Identical on every machine."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    dt = getattr(torch, dtype)
    if kind == "normal":
        t = torch.randn(*shape, generator=g, dtype=torch.float32)
    elif kind == "uniform":
        t = torch.rand(*shape, generator=g, dtype=torch.float32)
    elif kind == "wide":
        # Deliberately wide dynamic range: this is what makes summation order
        # observable.  Small terms are lost or kept depending on the order.
        t = torch.randn(*shape, generator=g, dtype=torch.float32)
        exps = torch.randint(-12, 13, t.shape, generator=g)
        t = t * torch.pow(torch.tensor(2.0), exps.float())
    else:
        raise ValueError(kind)
    return (t * scale).to(dt)


# ---------------------------------------------------------------- arithmetic

@probe("acc.reduction_order", "Summation order",
       "accumulation",
       "A sum over many terms lands on a different value.  Losses drift apart "
       "over training steps while every individual step looks fine.")
def p_reduction_order(dev, torch):
    x = _gen((1 << 20,), "float32", seed=1, kind="wide").to(dev)
    return {
        "sum": x.sum(),
        "sum_reversed": x.flip(0).sum(),
        "sum_dim_split": x.reshape(1024, 1024).sum(dim=1).sum(),
        "mean": x.mean(),
        "cumsum_tail": x[:4096].cumsum(0)[-1],
    }


@probe("acc.fp16_accumulate", "Half-precision accumulator width",
       "accumulation",
       "A matmul in fp16 accumulates in fp16 instead of fp32.  Long reductions "
       "quietly lose the small terms; the model trains but converges worse.")
def p_fp16_accumulate(dev, torch):
    a = _gen((256, 4096), "float16", seed=2, scale=0.05).to(dev)
    b = _gen((4096, 256), "float16", seed=3, scale=0.05).to(dev)
    return {"matmul_fp16": (a @ b).float()}


@probe("acc.fp32_matmul", "float32 matmul precision",
       "precision",
       "A float32 matmul does not carry the precision that float32 implies, "
       "usually because the hardware substitutes a shorter mantissa.  Results "
       "look right to three digits and wrong to seven.")
def p_fp32_matmul(dev, torch):
    ah = _gen((512, 512), "float32", seed=4)
    bh = _gen((512, 512), "float32", seed=5)
    a, b = ah.to(dev), bh.to(dev)
    out = {"matmul_fp32": a @ b, "addmm": torch.addmm(a[:, :512], a, b)}
    # How far is this device's float32 matmul from the exact answer?  The same
    # product computed in float64 on the CPU is a yardstick belonging to neither
    # device, so each one can be measured against it on its own terms.
    exact = ah.double() @ bh.double()
    err = float((out["matmul_fp32"].detach().to("cpu").double() - exact).abs().max())
    scale = float(exact.abs().max())
    out["error_vs_exact_info"] = torch.tensor(err)
    out["mantissa_bits_info"] = torch.tensor(_mantissa_bits(err, scale))
    return out


def _mantissa_bits(err, scale):
    """Turn an absolute error into the number of mantissa bits it implies.

    float32 carries 24 bits, TensorFloat-32 carries 11, bfloat16 carries 8.  A
    matmul accumulates error across K terms, so this is a floor rather than an
    exact count, but the distance between 24 and 11 is far too wide to mistake.
    """
    if err <= 0 or scale <= 0:
        return 24.0
    return max(0.0, -math.log(err / scale, 2))


@probe("acc.bf16_matmul", "bfloat16 support and accumulator",
       "precision",
       "bfloat16 is unsupported or accumulates differently; mixed-precision "
       "recipes tuned elsewhere do not transfer.")
def p_bf16_matmul(dev, torch):
    a = _gen((256, 1024), "bfloat16", seed=6, scale=0.1).to(dev)
    b = _gen((1024, 256), "bfloat16", seed=7, scale=0.1).to(dev)
    return {"matmul_bf16": (a @ b).float()}


# ---------------------------------------------------------------- math library

@probe("lib.transcendental", "Transcendental function implementations",
       "math library",
       "exp, tanh, erf and rsqrt come from a different math library.  Every "
       "activation in the network starts from a slightly different place.")
def p_transcendental(dev, torch):
    x = (_gen((1 << 16,), "float32", seed=8, scale=8.0)).to(dev)
    out = {
        "exp": torch.exp(x.clamp(-20, 20)),
        "tanh": torch.tanh(x),
        "erf": torch.erf(x),
        "rsqrt": torch.rsqrt(x.abs() + 1e-6),
        "log1p": torch.log1p(x.abs()),
        "sigmoid": torch.sigmoid(x),
    }
    try:
        out["gelu"] = torch.nn.functional.gelu(x)
        out["gelu_tanh"] = torch.nn.functional.gelu(x, approximate="tanh")
    except Exception:
        pass
    return out


@probe("lib.softmax", "Softmax numerics",
       "math library",
       "Softmax over large-magnitude logits is stabilised differently.  "
       "Attention weights differ before any weight has changed.")
def p_softmax(dev, torch):
    x = _gen((64, 2048), "float32", seed=9, scale=30.0).to(dev)
    return {
        "softmax": torch.softmax(x, dim=-1),
        "logsumexp": torch.logsumexp(x, dim=-1),
        "softmax_fp16": torch.softmax(x.half(), dim=-1).float(),
    }


@probe("lib.normalisation", "Normalisation epsilon placement",
       "math library",
       "layer_norm places epsilon inside or outside the square root.  On "
       "near-constant inputs the outputs separate sharply.")
def p_normalisation(dev, torch):
    x = _gen((128, 1024), "float32", seed=10, scale=1e-3).to(dev)
    x = x + 1.0  # near-constant: variance is tiny, epsilon dominates
    w = _gen((1024,), "float32", seed=11).to(dev)
    b = _gen((1024,), "float32", seed=12).to(dev)
    out = {"layer_norm": torch.nn.functional.layer_norm(x, (1024,), w, b, 1e-5)}
    try:
        rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        out["rms_norm"] = rms * w
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- ordering

@probe("ord.atomics", "Atomic accumulation order",
       "parallel ordering",
       "Scatter-adds with repeated indices land in a hardware-dependent order. "
       "Gradient accumulation becomes irreproducible.")
def p_atomics(dev, torch):
    import torch as _t
    g = _t.Generator(device="cpu").manual_seed(77)
    idx = _t.randint(0, 64, (1 << 16,), generator=g)
    src = _gen((1 << 16,), "float32", seed=13, kind="wide")
    out = _t.zeros(64, dtype=_t.float32)
    r = {"index_add_fp32": out.to(dev).index_add_(0, idx.to(dev), src.to(dev))}
    try:
        # Scaled so the running total stays inside the fp16 range: the question
        # is what order the additions happen in, not what happens at overflow.
        small = (src / src.abs().max() * 0.01).half()
        o16 = _t.zeros(64, dtype=_t.float16).to(dev)
        r["index_add_fp16"] = o16.index_add_(0, idx.to(dev), small.to(dev)).float()
    except Exception:
        pass
    return r


@probe("ord.nondeterminism", "Run-to-run stability",
       "parallel ordering",
       "The same kernel on the same device gives two different answers.  No "
       "comparison against any other machine can be trusted until this is known.",
       severity="critical", cross_device=False)
def p_nondeterminism(dev, torch):
    a = _gen((512, 512), "float32", seed=14).to(dev)
    b = _gen((512, 512), "float32", seed=15).to(dev)
    first = a @ b
    second = a @ b
    third = (a @ b) + 0.0
    return {"run1": first, "run2": second, "run3": third}


@probe("ord.reduction_shape", "Reduction over awkward shapes",
       "parallel ordering",
       "Reductions along a non-contiguous or ragged axis take a different "
       "code path and a different order.")
def p_reduction_shape(dev, torch):
    x = _gen((97, 1021), "float32", seed=16, kind="wide").to(dev)
    xt = x.t()
    return {
        "sum_rows": x.sum(dim=1),
        "sum_cols": x.sum(dim=0),
        "sum_transposed": xt.sum(dim=0),
        "norm": x.norm(),
        "var": x.var(dim=1, unbiased=False),
    }


# ---------------------------------------------------------------- randomness

@probe("rng.device_stream", "Device random number stream",
       "randomness",
       "The same seed produces a different stream of numbers.  Initialisation, "
       "dropout and sampling all diverge from step zero.",
       severity="critical", expect="bitwise")
def p_rng_device(dev, torch):
    torch.manual_seed(20260101)
    out = {}
    try:
        out["randn"] = torch.randn(4096, device=dev)
        out["rand"] = torch.rand(4096, device=dev)
        torch.manual_seed(20260101)
        out["randn_reseeded"] = torch.randn(4096, device=dev)
    except Exception as exc:
        raise
    try:
        x = torch.ones(4096, device=dev)
        torch.manual_seed(7)
        out["dropout"] = torch.nn.functional.dropout(x, p=0.5, training=True)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- selection

@probe("sel.tie_breaking", "Tie breaking in sort, topk and argmax",
       "selection",
       "Equal values are ordered differently.  Greedy decoding picks a "
       "different token; evaluation metrics shift with no model change.",
       expect="bitwise")
def p_tie_breaking(dev, torch):
    import torch as _t
    g = _t.Generator(device="cpu").manual_seed(31)
    x = _t.randint(0, 8, (64, 512), generator=g).float().to(dev)
    out = {
        "argmax": x.argmax(dim=-1).float(),
        "topk_idx": x.topk(8, dim=-1).indices.float(),
        "sort_idx": x.sort(dim=-1).values,
    }
    try:
        out["unique"] = _t.unique(x[0]).float()
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- composite

@probe("net.attention", "Scaled dot-product attention",
       "fused kernel",
       "The fused attention kernel differs from the unfused one, or is absent. "
       "This is the single most common place for a port to go quietly wrong.",
       severity="critical")
def p_attention(dev, torch):
    q = _gen((2, 8, 128, 64), "float32", seed=17, scale=0.5).to(dev)
    k = _gen((2, 8, 128, 64), "float32", seed=18, scale=0.5).to(dev)
    v = _gen((2, 8, 128, 64), "float32", seed=19, scale=0.5).to(dev)
    scale = 1.0 / math.sqrt(64)
    manual = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1) @ v
    out = {"attention_manual": manual}
    try:
        fused = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out["attention_fused"] = fused
        out["fused_vs_manual_info"] = (fused - manual).abs().max()
    except Exception:
        pass
    try:
        out["attention_causal"] = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)
    except Exception:
        pass
    return out


@probe("net.conv", "Convolution algorithm selection",
       "fused kernel",
       "A different convolution algorithm is chosen, with a different "
       "summation order inside it.")
def p_conv(dev, torch):
    x = _gen((4, 16, 32, 32), "float32", seed=20).to(dev)
    w = _gen((16, 16, 3, 3), "float32", seed=21, scale=0.1).to(dev)
    return {"conv2d": torch.nn.functional.conv2d(x, w, padding=1)}


@probe("mem.layout", "Memory layout and striding",
       "memory",
       "Non-contiguous and channels-last tensors take a different path, and "
       "the results do not agree with the contiguous one.")
def p_layout(dev, torch):
    x = _gen((8, 32, 16, 16), "float32", seed=22).to(dev)
    xc = x.contiguous(memory_format=torch.channels_last)
    return {
        "contig_sum": x.sum(dim=(2, 3)),
        "channels_last_sum": xc.sum(dim=(2, 3)),
        "permuted_sum": x.permute(0, 2, 3, 1).sum(dim=(1, 2)),
        "strided_slice": x[:, ::2, ::2, :].sum(dim=(1, 2)),
    }


def run_all(backend, only=None, verbose=False):
    """Run every probe on one device and fingerprint the result."""
    import torch
    from .fingerprint import fingerprint

    dev = backend.device
    results = {}
    for entry in REGISTRY:
        pid = entry["id"]
        if only and not any(pid.startswith(o) or o == pid for o in only):
            continue
        record = {k: entry[k] for k in ("id", "title", "layer", "symptom",
                                        "severity", "expect", "cross_device")}
        try:
            tensors = entry["fn"](dev, torch)
            backend.synchronize()
            record["tensors"] = {}
            for name, t in tensors.items():
                if not hasattr(t, "detach"):
                    t = torch.tensor(t)
                record["tensors"][name] = fingerprint(t)
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "error"
            record["error"] = "%s: %s" % (type(exc).__name__, exc)
            record["tensors"] = {}
        if verbose:
            print("  %-26s %s" % (pid, record["status"]))
        results[pid] = record
    return results
