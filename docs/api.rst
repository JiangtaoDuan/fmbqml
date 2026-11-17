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

Both model classes provide ``get_data()``, ``set_data()``, and
``set_parameters()`` for accessing or updating model inputs and settings.
Updates are validated before the model is mutated. A successful update clears
derived caches, while previously returned result objects remain independent
snapshots.

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

The former model-level accessors and reporting methods
``get_test_results()``, ``get_joint_results()``,
``get_joint_test_results()``, ``get_breakpoint()``, ``get_breakpoints()``,
``is_rejected()``, and model-level summary or plotting methods are not the
current public access pattern. Capture and use the result object returned
by each analysis method instead.
