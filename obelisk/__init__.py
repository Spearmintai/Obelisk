"""Obelisk: environment alignment with a reference baseline.

The unit of work is an *assumption*: something a program believes about its
accelerator that is never written down in the program itself.  Obelisk
enumerates those assumptions, measures each one as a numerical fingerprint on a
reference device, carries the fingerprints to a target device in a portable
blueprint, and reports which assumptions survived the move.
"""

__version__ = "0.1.0"

from .fingerprint import fingerprint, compare_fingerprints  # noqa: F401
