
"""Public interface for structural-break analysis in factor models."""

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
