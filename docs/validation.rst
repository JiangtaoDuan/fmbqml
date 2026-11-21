Validation and support policy
=============================

The model constructors and public model methods validate user-supplied panel
data and public options. ``set_data()``, ``set_parameters()``, and direct
assignment of configuration attributes validate proposed changes before
mutating model state. ``get_parameters()`` returns the current configuration. A successful update clears
derived model caches, while previously returned result objects remain
independent snapshots.

Both model classes accept ``trim_ratio``, ``max_factors``, and
``factor_criterion`` in that positional order or as keyword arguments.
``MultiBreakQML`` also accepts ``max_break`` as a fourth positional argument
or by keyword. Earlier versions used ``max_break`` as its first positional
argument; use ``max_break=...`` when updating those calls.

Lower-level functions in package submodules may be called directly, but they
are algorithmic building blocks rather than a second model-level validation
interface. Direct callers are responsible for satisfying each function's
mathematical and structural input requirements. Lower-level routines retain
checks needed for their own calculations and numerical stability, such as
matrix dimensions, feasible segment lengths, rank conditions, and
log-determinant evaluation.
