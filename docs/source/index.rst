.. Merlin documentation master file, created by sphinx-quickstart on Mon Jul  4 11:20:05 2022.
   You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

Welcome to Merlin's documentation!
==================================

.. raw:: latex

   \chapter{Introduction}

Merlin is a C++20 / `CUDA <https://docs.nvidia.com/cuda/index.html>`_ library for compressing and interpolating
few-group homogenized cross sections from lattice calculations in the two-step approach :cite:p:`galia2020dynamic`. The
library also supports a Python interface facilitated by `Pybind11 <https://pybind11.readthedocs.io/en/stable/>`_.

Major features include:

-  **Multidimensional array**: support multidimensional array of double precision on CPU, GPU (CUDA) and out-of-core.

-  **CUDA RAII classes**: wrap CUDA functions in class constructors and destructors to avoid resource leakage.

-  **Multi-variate interpolation**: interpolate data over a multidimensional Cartesian grids by linear or polynomial
   interpolation.

-  **Polynomial regression**: model a data using monomials.

-  **Tensor decomposition**: decompose multidimensional array into sum of tensor products of one-dimensional vectors by
   the gradient-based CANDECOMP-PARAFAC method :cite:p:`acar2011scalable`.

-  **Parallelism**: reduce calculation time by exploiting the multithread of CPU with OpenMP and GPU with CUDA kernels.

.. raw:: html

   <h2>Where to go from here ?</h2>

.. grid:: 3
   :gutter: 1
   :class-container: .container-lg

   .. grid-item-card:: Installation
      :shadow: md
      :text-align: center

      Follow the :doc:`installation` section to install the library.

   .. grid-item-card:: User guide
      :shadow: md
      :text-align: center

      A short guide on how to use Merlin is introduced in :doc:`userguide` section.

   .. grid-item-card:: C++ API
      :shadow: md
      :text-align: center

      All C++ classes and functions are summarized in :doc:`capi/index` section.

   .. grid-item-card:: Python API
      :shadow: md
      :text-align: center

      Python interface is documented in the :doc:`pyapi/index` section.

   .. grid-item-card:: Developer guide
      :shadow: md
      :text-align: center

      To extend the code for personal use, refer to the :doc:`developer/index` section.

   .. grid-item-card:: License
      :shadow: md
      :text-align: center

      The MIT License applies to this project, see :doc:`license`.

.. raw:: html

   <h2>Bibliography</h2>

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   userguide
   capi/index
   pyapi/index
   developer/index
   license

.. bibliography::
