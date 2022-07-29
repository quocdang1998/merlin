.. Merlin documentation master file, created by
   sphinx-quickstart on Mon Jul  4 11:20:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Merlin's documentation!
==================================

.. raw:: latex

   \chapter{Introduction}

Merlin is a package for multilinear interpolation of a large dataset. It
provides parallel construction and evaluation with out-of-core support for
processing a very large dataset. It employs the parallelism of GPU for
processing and evaluating basis functions from a representation of input data.

Merlin is written in C++, with a wrapper to Python through the package Cython.
All the heavy calculations and comminications with GPU or system memory are
performed in the C++ core of the package.

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   capi/index
   developper_guide

