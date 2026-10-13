"""The card library: what is known about devices nobody here has.

A full profile is most of a megabyte, because it carries the numerical
fingerprints needed to answer "does my code run *correctly*".  The question
"does my code run at all" needs far less: the list of operators that work, the
list that does not, and the data types that can compute.  That reduces to a few
kilobytes, which is small enough to ship with the tool.

So the library holds capability cards, not profiles.  Anyone who installs
Obelisk can check their code against every card that has ever been measured,
without owning any of the hardware.

A card is only true of the configuration it was taken on.  A driver update or a
framework upgrade can add or remove kernels, so every card records the versions
it was measured under and `obe check` says so when they differ from yours.  A
stale card is worse than no card, because it is believed.
"""

import json
import os

CARD_FORMAT = "obelisk-card/1"
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED = os.path.join(HERE, "cards")


def slug(device):
    """A stable file name for a device: vendor, model and backend."""
    name = (device.get("name") or "unknown").lower()
    for ch in " /\\(),.":
        name = name.replace(ch, "-")
    while "--" in name:
        name = name.replace("--", "-")
    return "%s-%s" % (name.strip("-"), device.get("backend", "?"))


def from_profile(profile):
    """Reduce a full profile to the part that answers 'will it run'."""
    payload = profile.get("payload") or {}
    cov = payload.get("coverage") or {}
    ops = cov.get("operators", [])

    def names(*statuses):
        return sorted({o["op"] for o in ops if o.get("status") in statuses})

    device = profile.get("device") or {}
    return {
        "format": CARD_FORMAT,
        "device": {k: device.get(k) for k in
                   ("name", "backend", "torch", "runtime_version", "driver",
                    "memory_gb", "capability", "arch")},
        "measured": profile.get("created"),
        "profile_digest": profile.get("digest"),
        "counts": cov.get("counts", {}),
        "missing": names("MISSING", "ERROR"),
        "wrong": names("MISMATCH", "SHAPE"),
        "exact": len([o for o in ops if o.get("status") == "EXACT"]),
        "close": len([o for o in ops if o.get("status") == "CLOSE"]),
        "dtypes": payload.get("dtypes", {}),
    }


def save(card, path):
    with open(path, "w") as fh:
        json.dump(card, fh, separators=(",", ":"), sort_keys=True)
    return path


def load(path):
    with open(os.path.expanduser(path)) as fh:
        return json.load(fh)


def bundled():
    """Every card shipped with this copy of Obelisk."""
    out = []
    if not os.path.isdir(BUNDLED):
        return out
    for fn in sorted(os.listdir(BUNDLED)):
        if not fn.endswith(".json"):
            continue
        try:
            card = load(os.path.join(BUNDLED, fn))
        except Exception:
            continue
        out.append({"name": fn[:-5], "path": os.path.join(BUNDLED, fn),
                    "card": card})
    return out


def resolve(reference):
    """Accept a file path, or the name of a card shipped with the tool."""
    expanded = os.path.expanduser(reference)
    if os.path.isfile(expanded):
        return expanded, None
    for entry in bundled():
        if entry["name"] == reference or reference.lower() in entry["name"]:
            return entry["path"], entry["name"]
    known = ", ".join(e["name"] for e in bundled()) or "none bundled"
    raise FileNotFoundError(
        "%s is neither a file nor a card shipped with Obelisk (have: %s)"
        % (reference, known))


def as_profile(card):
    """Present a card in the shape the checker expects of a profile."""
    operators = []
    for op in card.get("missing", []):
        operators.append({"op": op, "status": "MISSING"})
    for op in card.get("wrong", []):
        operators.append({"op": op, "status": "MISMATCH"})
    return {
        "device": card.get("device", {}),
        "digest": card.get("profile_digest"),
        "created": card.get("measured"),
        "payload": {"coverage": {"operators": operators,
                                 "counts": card.get("counts", {})},
                    "dtypes": card.get("dtypes", {})},
        "_card": True,
    }


def staleness(card, torch_version=None):
    """Say plainly when a card describes a different setup from this one."""
    notes = []
    d = card.get("device") or {}
    if torch_version and d.get("torch") and d["torch"] != torch_version:
        notes.append("this card was measured under torch %s; you are running %s"
                     % (d["torch"], torch_version))
    if d.get("driver"):
        notes.append("driver at the time of measurement: %s" % d["driver"])
    return notes
