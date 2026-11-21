API reference
=============

This page documents the public model and result classes. The supported
high-level interface is imported directly from ``fmbqml``.

.. currentmodule:: fmbqml

Model classes
-------------

.. autoclass:: SingleBreakQML
   :members:
   :show-inheritance:

.. autoclass:: MultiBreakQML
   :members:
   :show-inheritance:

Result classes
--------------

.. autoclass:: SingleBreakResults
   :members:
   :show-inheritance:

.. autoclass:: MultiBreakResults
   :members:
   :show-inheritance:

Main analysis methods
---------------------

The LR-test methods ``SingleBreakQML.lr_test()`` and
``MultiBreakQML.joint_sup_lr_test()`` accept ``n_jobs`` to control
parallel Monte Carlo simulation. The default ``n_jobs=-1`` uses all
available joblib workers; set ``n_jobs=1`` to use one worker or a positive
integer to limit concurrent workers. The number of simulations is separately
controlled by ``n_sim``. The accepted values for ``n_jobs`` are ``-1``
and positive integers.

``SingleBreakQML`` provides:

* ``lr_test()`` to test the null hypothesis of no structural break against
  the single-break alternative.
* ``estimate_breakpoint()`` to estimate a breakpoint under the assumption
  that one break is present, without making an LR-test decision.
* ``get_factor_estimates()`` to retrieve the estimated factors and loadings.

``MultiBreakQML`` provides:

* ``estimate_breaks_jointly()`` to jointly select the number and locations
  of breaks using the QML information criterion. Setting ``min_break=0``
  allows the no-break specification to be considered.
* ``joint_sup_lr_test()`` to test the no-break null against an alternative
  with a prespecified number of breaks.
* ``get_factor_estimates()`` to retrieve the estimated factors and loadings.

Both model classes provide ``factors_`` and ``factor_loadings_`` properties
for direct access to copies of the estimated factor and loading matrices.
These properties compute the factor estimates on first access and reuse the
model cache until the data or configuration changes.

Both model classes provide ``get_data()`` and ``set_data()`` for accessing
or replacing panel data, and ``get_parameters()`` and ``set_parameters()``
for reading or updating model configuration. The configuration attributes
can also be changed by direct assignment (e.g., ``model.trim_ratio = 0.2``).
All configuration updates are validated before the model is mutated and
clear derived caches on success. Invalid updates leave the previous configuration
and caches intact; previously returned result objects remain independent
snapshots. Panel data and sample dimensions cannot be replaced by assignment;
use ``set_data()`` instead.

Accessing factor estimates
--------------------------

The factor and loading matrices can be retrieved separately from either
model class::

   factors = model.factors_
   loadings = model.factor_loadings_

Both properties return independent NumPy array copies. Their shapes are
``(T, r)`` for the estimated common factors and ``(N, r)`` for the factor
loadings, where ``r`` is the selected number of factors. The complete
factor-estimation dictionary remains available through
``model.get_factor_estimates()``. Its display summarizes each array by shape
and a short sample; indexing the dictionary returns the complete array.


Exporting result dictionaries
-----------------------------

Both result classes provide ``to_dict(include_diagnostics=True)``. The
method returns an independent Python dictionary. By default, all stored
fields are included for backward compatibility. Set
``include_diagnostics=False`` to omit large diagnostic paths and arrays while
keeping the principal test or estimation results. For single-break results,
this omits ``candidate_breaks``, ``lr_profile``, and ``qml_profile``. For
multiple-break results, it omits ``ic_path`` and ``simulated_statistics``.

The returned dictionary displays one top-level field per line::

   print(result.to_dict(include_diagnostics=False))

It is a dictionary subclass and can be indexed and modified as usual.


Interpreting breakpoint results
-------------------------------

For a single-break LR test, ``candidate_breakpoint`` is the location that
maximizes the LR profile under the alternative hypothesis (H1). The
``breakpoint`` attribute is populated only when the null hypothesis (H0) of
no structural break is rejected; otherwise it is ``None``. The candidate
remains available even when H0 is not rejected.

For a joint sup-LR test, ``candidate_breakpoints`` stores the locations
selected under the prespecified multiple-break alternative (H1).
``breakpoints`` stores the accepted locations only when H0 is rejected;
otherwise it is an empty list. Candidate locations should therefore not be
interpreted as statistically accepted breaks when the test does not reject H0.

The compact representation produced by ``print(result)`` reports the test
decision and accepted breakpoint location(s) only. If the null hypothesis is
not rejected, it reports ``Accepted breakpoint: None`` for a single-break
test or ``Accepted breakpoints: None`` for a joint test. It does not present
the candidate location(s) as accepted breaks.

Calling ``result.summary()`` provides the complete statistical report.
For LR tests, the summary includes the candidate breakpoint location(s)
selected under H1 and the accepted breakpoint location(s) determined by the
test decision. Candidate locations remain visible in the summary even when
the null hypothesis is not rejected. The ``summary()`` method returns a
structured report object that can be displayed directly or inspected
programmatically.

Result objects also provide profile plotting methods where applicable.
``SingleBreakResults.plot_lr_profile()`` plots the saved LR profile for an
LR-test result, and ``plot_qml_profile()`` plots the saved single-break QML
profile. ``MultiBreakResults.plot_mqml_profile()`` displays the saved joint
QML information-criterion path. These methods use stored results and do not
rerun the estimation.

Data labels and breakpoint indexing
------------------------------------

Breakpoint locations are reported using a one-based observation position.
A breakpoint ``k`` separates observations ``1, ..., k`` from observations
``k + 1, ..., T``. When a pandas DataFrame is supplied, the original row
index is preserved; the result objects expose the corresponding label
alongside the integer breakpoint position when available.

Model configuration
-------------------

Both models accept ``max_factors`` and ``factor_criterion``. The supported
Bai--Ng criteria are ``IC1``, ``IC2``, ``IC3``, ``PC1``, ``PC2``, ``PC3``,
``AIC3``, and ``BIC3``; integer indices 0 through 7 are also accepted.

``SingleBreakQML`` additionally accepts ``trim_ratio`` (default 0.30).
``MultiBreakQML`` accepts ``max_break`` (default 5) and ``trim_ratio``
(default 0.10). The trimming ratio determines the admissible breakpoint
search region or minimum regime length used by the corresponding procedure.

Access analysis results through the result object returned by each analysis
method. Summary and plotting methods are also available on that object.
