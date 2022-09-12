.. Merlin documentation master file, created by
   sphinx-quickstart on Mon Jul  4 11:20:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Merlin's documentation!
==================================

.. raw:: latex

   \chapter{Introduction}

Merlin is a package written in C++, with Python wrapper package made possible by
`Cython <https://cython.readthedocs.io/en/latest/>`_ for processing, evaluating
and interpolating multidimensional dataset . It fastens calculations by
exploiting the parallelism of CPU and HPC system equipped with multiple GPUs.
The tool-kit can be scalable to big dataset thanks to its support for
thread-safe out-of-core array.

.. raw:: html

   <h2>Where to go from here ?</h2>

.. panels::
   :container: .container-lg
   :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2
   :card: shadow
   :header: text-center
   :body: text-center

   Installation
   ^^^^^^^^^^^^

   To install Merlin, follow the :doc:`installation`.
   ---

   C++ API
   ^^^^^^^

   To see the code documentation, see the :doc:`capi/index`.
   ---

   Developer guide
   ^^^^^^^^^^^^^^^^

   To extend the code for personal use, see the :doc:`developer/index`.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   capi/index
   pyapi/index
   developer/index

