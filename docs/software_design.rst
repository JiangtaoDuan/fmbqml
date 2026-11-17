Software design
===============

The package separates the user-facing model classes from the computational
routines used by the estimation and testing procedures. The principal public
classes are ``SingleBreakQML`` and ``MultiBreakQML``. Their corresponding
result classes, ``SingleBreakResults`` and ``MultiBreakResults``, provide
attribute-based access, reporting, and visualization.

Package organization
--------------------

.. code-block:: text

   fmbqml/
   ├── single_break/       # Single-break model, QML profile, and results
   ├── multiple_break/     # Joint estimation, testing, optimization, and results
   ├── factor_tools/       # Factor-number selection and factor estimation
   ├── statistical_tools/  # HAC estimation and critical-value simulation
   ├── break_tools/        # Structural-break type classification
   └── utils/              # Validation, criteria, partitions, plotting, formatting

The model classes coordinate these components and expose the main analysis
workflows. Lower-level functions are implementation building blocks; the
model constructors and public model methods are the primary boundary for
validating user-supplied panel data and public options.

Numerical safeguards
--------------------

The implementation includes numerical safeguards for covariance-matrix
log-determinants and for the construction of HAC covariance estimates. These
safeguards are intended to improve numerical stability without changing the
statistical definitions of the procedures.
