"""Public interface for structural-break analysis in factor models."""

from .single_break.single_break_qml import SingleBreakQML
from .multiple_break.multiple_break_qml import MultiBreakQML

__all__ = [
    "SingleBreakQML",
    "MultiBreakQML",
]
