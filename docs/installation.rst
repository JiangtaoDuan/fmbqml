Installation
============

Requirements
------------

* Python 3.8 or later.
* NumPy, SciPy, pandas, Matplotlib, and joblib. These dependencies are
  installed automatically when ``fmbqml`` is installed with pip.

Install from PyPI
-----------------

.. code-block:: console

   python -m pip install fmbqml==0.1.6

To install the latest published release instead, omit the version pin:

.. code-block:: console

   python -m pip install fmbqml

Install from a source checkout
------------------------------

From the root directory of the cloned repository, run:

.. code-block:: console

   python -m pip install .

Development installation
------------------------

To install the package together with the test and documentation tools, run:

.. code-block:: console

   python -m pip install -e ".[test,docs]"

Then run the tests with ``python -m pytest`` or build the documentation with
``python -m sphinx -b html docs docs/_build/html``.
