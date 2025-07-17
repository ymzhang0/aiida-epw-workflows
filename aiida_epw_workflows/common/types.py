# -*- coding: utf-8 -*-
"""Module with common data types."""
from enum import Enum

class RestartType(Enum):
    """Enumeration of ways to restart a calculation."""

    FROM_SCRATCH = 'from_scratch'
    FROM_EPB = 'from_epb_file'
    FROM_EPMATWP = 'from_epmatwp'
    FROM_EPHMAT = 'from_ephmat'
