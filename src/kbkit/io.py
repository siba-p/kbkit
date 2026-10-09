"""Read RDFs from GROMACS ``gmx rdf`` .xvg output or plain two-column text."""
from __future__ import annotations

import re

import numpy as np


def read_rdf(path, column: int = 1):
    """Return (r, g) from an .xvg / text file; '#', '@' and '&' lines are skipped."""
    r, g = [], []
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if not t or t[0] in "#@&":
                continue
            p = re.split(r"[\s,]+", t)
            try:
                r.append(float(p[0]))
                g.append(float(p[column]))
            except (ValueError, IndexError):
                continue
    return np.asarray(r), np.asarray(g)
