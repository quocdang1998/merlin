.. Merlin documentation master file, created by sphinx-quickstart on Mon Jul  4 11:20:05 2022.
   You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

Welcome to Merlin's documentation!
==================================

.. raw:: latex

   \chapter{Introduction}

Merlin is a C++ / `CUDA <https://docs.nvidia.com/cuda/index.html>`_ library for processing and interpolation of
multi-parameterized cross sections resulting from lattice calculations in the two-step approach
:cite:p:`galia2020dynamic`. The library also supports a Python interface facilitated by
`Cython <https://cython.readthedocs.io/en/latest/>`_.

Major features include:

-  **Multidimensional array**: supports multidimensional array of double precision on CPU, GPU (CUDA) and out-of-core.

-  **Polynomial interpolation**: interpolate a dataset over a multidimensional Cartesian grid and hierarchical grids
   :cite:p:`garcke2006sparse` by Lagrange :cite:p:`berrut2004barycentric` and Newton method
   :cite:p:`neidinger2019multivariate`.

-  **Tensor decomposition**: decompose multidimensional array to sum of tensor products of one-dimensional vectors by
   gradient approach :cite:p:`acar2011scalable`.

-  **Parallelism**: accelerate the calculation of coefficients and evaluation of the interpolation by transporting the
   workload to GPUs.

.. raw:: html

   <h2>Where to go from here ?</h2>

.. grid:: 3
   :gutter: 1
   :class-container: .container-lg

   .. grid-item-card:: Installation
      :shadow: md
      :text-align: center

      To install Merlin, follow the :doc:`installation`.

   .. grid-item-card:: C++ API
      :shadow: md
      :text-align: center

      To see the code documentation, see the :doc:`capi/index`.

   .. grid-item-card:: Python API
      :shadow: md
      :text-align: center

      To see the code documentation, see the :doc:`pyapi/index`.

   .. grid-item-card:: Developer guide
      :shadow: md
      :text-align: center

      To extend the code for personal use, see the :doc:`developer/index`.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   capi/index
   pyapi/index
   developer/index

.. bibliography::
