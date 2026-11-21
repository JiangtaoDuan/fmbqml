User guide
==========

Input data
----------

Both model classes accept a finite numeric NumPy array or pandas DataFrame
with shape ``(T, N)``. Rows represent time observations and columns represent
cross-sectional units or variables. Missing and infinite values are not
supported. The package does not automatically demean or standardize the
input; apply transformations appropriate to the application before fitting
the model.

The model constructors expose the following main settings:

* ``max_factors``: maximum number of pseudo-factors considered (default 10).
* ``factor_criterion``: Bai--Ng criterion, specified as one of ``IC1``,
  ``IC2``, ``IC3``, ``PC1``, ``PC2``, ``PC3``, ``AIC3``, or ``BIC3``; integer
  indices 0 through 7 are also accepted.
* ``trim_ratio``: trimming fraction that determines the minimum segment
  length. The default is 0.30 for ``SingleBreakQML`` and 0.10 for
  ``MultiBreakQML``.
* ``max_break``: maximum number of breaks considered by ``MultiBreakQML``
  (default 5).

Breakpoint locations use a one-based observation convention. A breakpoint
``k`` separates observations ``1, ..., k`` from observations
``k + 1, ..., T``. If a pandas DataFrame is supplied, its row index is
preserved and can be used to associate estimated locations with dates.

Single-break analysis
---------------------

The following example illustrates the main workflow. The data-generating
process is intentionally simple; see the paper and the separate replication
materials for the manuscript's simulation design.

.. code-block:: python

   import numpy as np
   from fmbqml import SingleBreakQML

   rng = np.random.default_rng(2024)
   T, N, r = 300, 100, 3
   true_break = 100

   factors = rng.normal(size=(T, r))
   loadings_before = rng.normal(size=(N, r))
   loadings_after = loadings_before + rng.normal(size=(N, r))
   errors = rng.normal(scale=np.sqrt(r), size=(T, N))

   X = np.empty((T, N))
   X[:true_break] = factors[:true_break] @ loadings_before.T + errors[:true_break]
   X[true_break:] = factors[true_break:] @ loadings_after.T + errors[true_break:]

   model = SingleBreakQML(
       X, trim_ratio=0.15, max_factors=10, factor_criterion="IC2"
   )
   result = model.lr_test(
       alpha=0.05, classify=True, n_sim=1000, random_state=12345
   )

   print(result)             # concise overview
   print(result.summary())   # detailed statistical report

The single-break LR test evaluates the null hypothesis

.. math::

   H_0: \text{no structural break}

against the alternative that a single structural break is present. The test
searches over the admissible breakpoint locations and compares the maximum
LR statistic with a simulated critical value at the selected significance
level.

The candidate location under the alternative is available as
``result.candidate_breakpoint`` regardless of the test decision. If the null
hypothesis is rejected, the accepted breakpoint is available as
``result.breakpoint``. If the null is not rejected,
``result.breakpoint`` is ``None``: the candidate is still a location selected
under H1, but it is not an accepted break.

To estimate a breakpoint under the assumption that one exists, without
conducting a significance test, use ``model.estimate_breakpoint()``. This
direct QML estimation procedure is distinct from the LR test decision.

No-break interpretation
-----------------------

The no-break case is the null specification in the single-break LR test.
Failure to reject this null means that the test does not provide sufficient
evidence of a structural break at the chosen significance level. It does
not prove that the data contain no structural change.

For multiple-break QML estimation, ``estimate_breaks_jointly()`` uses
``min_break=0`` by default. This allows the information-criterion search to
select the no-break specification rather than requiring at least one break.
If the application requires at least one break, set ``min_break`` to a
positive integer and interpret the result conditional on that restriction.

Multiple-break analysis
-----------------------

``MultiBreakQML`` provides joint selection of the number and locations of
breaks using an information criterion, as well as joint sup-LR testing for
a prespecified number of breaks.

.. code-block:: python

   from fmbqml import MultiBreakQML

   model = MultiBreakQML(
       X, max_break=5, max_factors=10,
       factor_criterion="IC2", trim_ratio=0.10
   )

   # Jointly select the number and locations of breaks.
   estimate = model.estimate_breaks_jointly(min_break=0, classify=True)
   print(estimate.summary())
   print(estimate.format_ic_path_table())

   # Alternatively, test a prespecified number of breaks.
   test = model.joint_sup_lr_test(
       n_breaks=2, alpha=0.05, n_sim=2000,
       random_state=12345, classify=True
   )
   print(test.summary())

For joint LR testing, ``result.candidate_breakpoints`` stores the locations
selected under the prespecified alternative with ``n_breaks`` breaks.
``result.breakpoints`` stores the accepted breakpoints and is empty when
the null is not rejected. As in the single-break test, candidate locations
under H1 should not be interpreted as accepted breaks when the test does
not reject H0.

The joint QML estimation procedure and the joint sup-LR test answer
different questions. Joint QML estimation selects a break configuration
using the information criterion, whereas the joint sup-LR test evaluates
a null hypothesis against an alternative with a prespecified number of
breaks.

Result objects and reproducibility
----------------------------------

The model methods return result objects. Use attribute access to retrieve
statistics and estimated locations, ``summary()`` for a detailed report,
and ``to_dict()`` when an independent dictionary copy is needed. The
``to_dict(include_diagnostics=False)`` option omits large diagnostic paths
and arrays while retaining the main results. The returned dictionary displays
one top-level field per line when printed::

   print(result.to_dict(include_diagnostics=False))

Result objects are snapshots: subsequent model updates do not alter results
already returned to the caller. Both model classes also provide ``factors_``
and ``factor_loadings_`` properties for direct access to copies of the
estimated factor and loading matrices.

The LR critical values are approximated by Monte Carlo simulation. Set
``random_state`` to an integer to make the simulation reproducible. The
``n_sim`` argument controls the number of replications and therefore the
accuracy and computational cost of the approximation.
