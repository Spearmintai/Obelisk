"""A small, self-contained transformer used as the subject under test.

Nothing here is interesting as a model.  What matters is that it is written in
plain PyTorch with no custom kernels, its weights come from a fixed CPU seed, and
every intermediate tensor can be captured.  That makes it a fair question to ask
of two different accelerators: given identical weights and identical inputs,
where do you start to disagree, and how fast does the disagreement grow?
"""

import math


def build(torch, dim=256, heads=4, layers=4, vocab=512, seed=1234):
    nn = torch.nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.ln2 = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, 4 * dim)
            self.fc2 = nn.Linear(4 * dim, dim)

        def forward(self, x):
            B, T, C = x.shape
            h = self.ln1(x)
            q, k, v = self.qkv(h).split(C, dim=2)
            q = q.view(B, T, heads, C // heads).transpose(1, 2)
            k = k.view(B, T, heads, C // heads).transpose(1, 2)
            v = v.view(B, T, heads, C // heads).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // heads))
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
            att = torch.softmax(att, dim=-1)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            x = x + self.proj(y)
            h = self.ln2(x)
            x = x + self.fc2(torch.nn.functional.gelu(self.fc1(h)))
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.pos = nn.Parameter(torch.zeros(1, 128, dim))
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.lnf = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)

        def forward(self, idx):
            B, T = idx.shape
            x = self.emb(idx) + self.pos[:, :T]
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.lnf(x))

    # Weights are drawn on the CPU so that both devices start from identical bits.
    torch.manual_seed(seed)
    model = Model()
    return model


def make_batch(torch, batch=4, seqlen=64, vocab=512, seed=99):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, vocab, (batch, seqlen + 1), generator=g)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def forward_trace(backend, dim=256, layers=4, capture_every=True):
    """One forward pass, fingerprinting the output of every submodule."""
    import torch
    from .fingerprint import fingerprint

    dev = backend.device
    model = build(torch, dim=dim, layers=layers).to(dev).eval()
    x, _ = make_batch(torch)
    x = x.to(dev)

    trace = {}
    order = []
    handles = []

    def hook(name):
        def fn(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if hasattr(t, "detach"):
                trace[name] = fingerprint(t)
                order.append(name)
        return fn

    if capture_every:
        for name, mod in model.named_modules():
            if name and not list(mod.children()):
                handles.append(mod.register_forward_hook(hook(name)))

    with torch.no_grad():
        logits = model(x)
    backend.synchronize()
    for h in handles:
        h.remove()

    trace["__output__"] = fingerprint(logits)
    order.append("__output__")
    return {"order": order, "layers": trace}


def train_trace(backend, steps=50, lr=0.1, dim=256, layers=4):
    """A short training run.  Records the loss and the parameter state at every
    step, which is where a difference too small to see in one forward pass
    becomes a difference you can read off a chart."""
    import torch
    from .fingerprint import fingerprint

    dev = backend.device
    model = build(torch, dim=dim, layers=layers).to(dev).train()
    x, y = make_batch(torch)
    x, y = x.to(dev), y.to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    losses = []
    checkpoints = {}
    for step in range(steps):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        backend.synchronize()
        losses.append(float(loss))
        if step in (0, 1, 2, 4, 9, 19, 49, steps - 1):
            flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
            checkpoints["step%04d" % step] = fingerprint(flat)

    return {"losses": losses, "checkpoints": checkpoints, "steps": steps, "lr": lr}
