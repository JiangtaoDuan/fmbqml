
"""Public interface for structural-break analysis in factor models."""

__version__ = "0.1.6"

from .single_break.model import (
    SingleBreakQML,
    SingleBreakResults,
)

from .multiple_break.model import (
    MultiBreakQML,
    MultiBreakResults,
)

__all__ = [
    "SingleBreakQML",
    "SingleBreakResults",
    "MultiBreakQML",
    "MultiBreakResults",
]
