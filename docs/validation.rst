Validation and support policy
=============================

The model constructors and public model methods validate user-supplied panel
data and public options. ``set_data()`` and ``set_parameters()`` validate
proposed changes before mutating model state. A successful update clears
derived model caches, while previously returned result objects remain
independent snapshots.

Lower-level functions in package submodules may be called directly, but they
are algorithmic building blocks rather than a second model-level validation
interface. Direct callers are responsible for satisfying each function's
mathematical and structural input requirements. Lower-level routines retain
checks needed for their own calculations and numerical stability, such as
matrix dimensions, feasible segment lengths, rank conditions, and
log-determinant evaluation.
