"""
Concept Drift Detection.

This module contains concept drift detection methods. The purpose of a drift detector is to raise
an alarm if the data distribution changes. A good drift detector method is the one that maximizes
the true positives while keeping the number of false positives to a minimum.

"""

from .new_d3 import D3

# Try to import other modules if available
__all__ = ["D3"]

try:
    from .dawidd import dawidd
    __all__.append("dawidd")
except ImportError:
    pass

try:
    from .shape_dd import ShapeDD
    __all__.append("ShapeDD")
except ImportError:
    pass
