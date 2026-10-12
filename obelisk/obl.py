"""The .obl blueprint: everything one device measured, in one portable file.

The format is JSON.  That is a deliberate choice: the file has to survive being
copied to a machine with no network, read by a tool that was not installed from
the same wheel, and opened by a person who wants to check a number by eye.
"""

import datetime
import hashlib
import json
import os

FORMAT = "obelisk/1"


def digest(blueprint):
    """A content address for everything the blueprint asserts.

    A report cites the digests it was derived from, so a reader can confirm that
    the report in their hand was produced from the measurements in theirs, and a
    withdrawn measurement can be named precisely rather than described.
    """
    body = json.dumps({"kind": blueprint.get("kind"),
                       "device": blueprint.get("device"),
                       "payload": blueprint.get("payload")},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf8")).hexdigest()[:32]


def new(kind, backend_info, payload):
    blueprint = {
        "format": FORMAT,
        "kind": kind,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "device": backend_info,
        "payload": payload,
    }
    blueprint["digest"] = digest(blueprint)
    return blueprint


def verify_digest(blueprint):
    """Has this blueprint been altered since it was written."""
    claimed = blueprint.get("digest")
    if not claimed:
        return None
    return claimed == digest(blueprint)


def save(blueprint, path):
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass
    with open(path, "w") as fh:
        json.dump(blueprint, fh, indent=1, sort_keys=False)
    return path


def load(path):
    with open(os.path.expanduser(path)) as fh:
        data = json.load(fh)
    if data.get("format") != FORMAT:
        raise ValueError("%s is not an Obelisk blueprint (format=%r)"
                         % (path, data.get("format")))
    return data


def describe(blueprint):
    d = blueprint.get("device", {})
    return "%s on %s (%s, torch %s)" % (
        d.get("name", "?"), d.get("host", "?"), d.get("backend", "?"), d.get("torch", "?"))
