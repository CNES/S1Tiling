.. _contribute:

======================================================================
Contribute to S1Tiling development
======================================================================

If you intend to contribute to s1tiling source code:

.. code-block:: bash

    git clone https://gitlab.orfeo-toolbox.org/s1-tiling/s1tiling
    cd stiling
    git checkout develop
    python -m pip install -r requirements-dev.txt

To run the default test suite:

.. code-block:: bash

    pytest

To compile and check the documentation:

.. code-block:: bash

    (cd docs && make html)
    firefox build/html/index.html


Git workflow
------------

Branches
++++++++

* ``master`` branch is meant to contain stable and released products
* ``develop`` branch is meant to aggregate developments until a stable and
  fully operational version is ready for distribution.
* :samp:`{feature}` are meant to be used to implement features.

Contributions
+++++++++++++

Contributions are expected to be made as `gitlab Merge Requests
<https://docs.gitlab.com/ee/user/project/merge_requests/>`_.

1. Clone S1Tiling on https://gitlab.orfeo-toolbox.org/
2. Work on a new branch forked from ``develop``
3. Push the new branch onto OTB gitlab server
4. Connect to :samp:`https://gitlab.orfeo-toolbox.org/{yourname}/s1tiling`, and
   click on *create a merge request* from the latest branch pushed.
5. Fill in the rationale...


.. todo:: What about a CLA?
