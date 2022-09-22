Installation
============

System requirements
-------------------

Before compiling the package, ensure that these prerequisites have been already
installed:

-  C++ compiler: GNU ``g++>=9.3.0`` on **Linux** or Visual Studio 2022
   ``cl.exe`` on **Windows**, with OpenMP enabled.

-  Cmake: ``cmake>=3.21.0`` (detect -std=c++17 flag on CUDA).

-  Build-tool: GNU ``make`` on **Linux** or ``ninja`` on **Windows** (Ninja
   extenstion of MSVC).

-  CUDA ``nvcc>=11.4`` (optional, required if GPU parallelization option is
   ``ON``).

The Python interface requires these additional packages:

-  ``Cython>=3.0.0a10`` (support enum class)

-  ``Numpy>1.17``

-  ``Scikit-build>=0.15.0`` (support cmake build)

To compile the documentation, install the following packages:

-  ``Doxygen``

-  ``Sphinx`` + extensions (``breathe``, ``sphinx_doxysummary``)

   .. code-block:: sh

      pip install breathe
      pip install git+https://github.com/quocdang1998/doxysummary.git


Compilation
-----------

The compilation composes of 2 stages: the C++ core by using CMake, and the
Python package by wrapping the former with Cython.

C++ core
^^^^^^^^

On Linux, to compile the C++ core, open a the terminal and execute:

.. code-block:: sh

   cmake --preset=linux .
   cd build
   make -j

On Windows, because MSVC pre-defines some enviroment variables, compilation
with Visual Studio application is strongly recommended. Inside the application:

1. Configure CMake: **Project** -> **Configure merlin**

   .. image:: _img/installation_Configure.png
      :width: 100%

2. Build: **Build** -> **Build All**

   .. image:: _img/installation_Build.png
      :width: 100%

.. note::

   It is possible to compile the package from the terminal (cmd or Powershell),
   but users are responsible for assuring that enviroment variables are
   correctly set before the compilation, depending on location and version of
   Visual Studio installed on their machine (see `Building on the command line
   <https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#path_and_environment>`_).

   **Any failure due to wrong settings of enviroment variables while compiling
   the package in a Windows terminal is not our responsibility**.

   .. code-block:: powershell

      cmake --preset=windows .
      cd build
      ninja

Python package
^^^^^^^^^^^^^^

To compile the Python interface, go back to the source directory of the package
and run:

.. code-block:: sh

   pip install .

Developpers or users unable to install the package with ``pip`` can install the
package in the source directory with:

.. code-block:: sh

   python setup.py build_ext --inplace


Build options
-------------

CMake build options
^^^^^^^^^^^^^^^^^^^

.. envvar:: MERLIN_CUDA

   Build C++ Merlin library with CUDA ``nvcc``.

   :Value: ``ON``, ``OFF``
   :Default: ``ON``

.. envvar:: MERLIN_LIBKIND

   Specify the kind of compiled C++ library.

   By default, compile dynamic library on Linux and static library on Windows.

   :Value: ``AUTO``, ``STATIC``, ``SHARED``
   :Default: ``AUTO``

.. envvar:: MERLIN_TEST

   Build test executable.

   :Value: ``ON``, ``OFF``
   :Default: ``OFF``

