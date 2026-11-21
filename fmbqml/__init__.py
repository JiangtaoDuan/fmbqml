
"""Public interface for structural-break analysis in factor models."""

__version__ = "0.1.7"

from .multiple_break.model import (
    MultiBreakQML,
    MultiBreakResults,
)
from .single_break.model import (
    SingleBreakQML,
    SingleBreakResults,
)

__all__ = [
    "MultiBreakQML",
    "MultiBreakResults",
    "SingleBreakQML",
    "SingleBreakResults",
]
