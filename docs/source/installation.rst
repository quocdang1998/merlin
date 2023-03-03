Installation
============

System requirements
-------------------

Before compiling the package, ensure that these prerequisites have already been
installed:

-  C++ compiler: GNU ``g++>=9.3.0`` on **Linux** or Visual Studio 2022
   ``cl.exe`` on **Windows**, with OpenMP enabled.

-  Cmake: ``cmake>=3.21.0`` (detect -std=c++17 flag on CUDA).

-  Build-tool: GNU ``make`` on **Linux** or ``ninja`` on **Windows** (Ninja
   extenstion of MSVC).

-  CUDA ``nvcc>=11.4`` (optional, required if GPU parallelization option is
   ``ON``).

The Python interface requires these additional packages:

-  |Cython|_ (support enum class)

-  |Numpy|_

-  |Jinja2|_

-  |packaging|_

.. |Cython| replace:: ``Cython>=3.0.0a10``
.. _Cython: https://pypi.org/project/Cython/#history
.. |Numpy| replace:: ``Numpy>1.17``
.. _Numpy: https://pypi.org/project/numpy/
.. |Jinja2| replace:: ``Jinja2``
.. _Jinja2: https://pypi.org/project/Jinja2/
.. |packaging| replace:: ``packaging``
.. _packaging: https://pypi.org/project/packaging/

To compile the documentation, install the following packages:

-  |Doxygen|_

-  |Sphinx|_ + extensions (|sphinx_rtd_theme|_, |sphinx_tabs|_,
   |sphinx_panels|_, |breathe|_ and |sphinx_doxysummary|_)

   .. code-block:: sh

      pip install --no-deps Sphinx sphinx_rtd_theme sphinx_tabs sphinx_panels breathe
      pip install --no-deps git+https://github.com/quocdang1998/doxysummary.git

.. |Doxygen| replace:: ``Doxygen>=1.8.5``
.. _Doxygen: https://doxygen.nl/download.html
.. |Sphinx| replace:: ``Sphinx``
.. _Sphinx: https://www.sphinx-doc.org/
.. |sphinx_rtd_theme| replace:: ``sphinx_rtd_theme``
.. _sphinx_rtd_theme: https://sphinx-rtd-theme.readthedocs.io/en/stable/
.. |sphinx_tabs| replace:: ``sphinx_tabs``
.. _sphinx_tabs: https://sphinx-tabs.readthedocs.io/en/latest/
.. |sphinx_panels| replace:: ``sphinx_panels``
.. _sphinx_panels: https://sphinx-panels.readthedocs.io/en/latest/
.. |breathe| replace:: ``breathe``
.. _breathe: https://breathe.readthedocs.io/en/latest/
.. |sphinx_doxysummary| replace:: ``sphinx_doxysummary``
.. _sphinx_doxysummary: https://doxysummary.readthedocs.io/en/latest/


Compilation
-----------

C++ and CUDA interface
^^^^^^^^^^^^^^^^^^^^^^

On Linux, to compile the C++ core, open a terminal and execute:

.. code-block:: sh

   cmake --preset=linux .
   cd build
   make -j

On Windows, because MSVC pre-defines environment variables, compilation with
Visual Studio application is strongly recommended. Inside the application:

1. Configure CMake: **Project** :fa:`caret-right` **Configure merlin**

   .. image:: _img/installation_Configure.png
      :width: 100%

2. Build: **Build** :fa:`caret-right` **Build All**

   .. image:: _img/installation_Build.png
      :width: 100%

.. note::

   It is possible to compile the package from the terminal (cmd or Powershell),
   but user are responsible for assuring that enviroment variables are correctly
   set before the compilation, depending on location and version of Visual
   Studio installed on the machine (see `Building on the command line
   <https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#path_and_environment>`_).

   **We do not take responsibility for any failure due to wrong settings of
   enviroment variables while compiling the package in a Windows terminal**.

   .. code-block:: powershell

      cmake --preset=windows .
      cd build
      ninja

Python package
^^^^^^^^^^^^^^

Before compiling the Python interface, **make sure that the C++/CUDA interface
have been compiled**.

If the package can be installed using ``pip``, go back
to the source directory (containing ``setup.py``) and run:

.. code-block:: sh

   pip install .

.. note::

   If ``setuptools>=30`` has been installed, build dependancies listed in the
   section :ref:`installation:System requirements` above are not required.
   ``setuptools`` will install automatically on the run (checkout
   `PEP 517 <https://peps.python.org/pep-0517/>`_).

If installation in the source directory is preferred (build dependancies must
have already been installed):

.. code-block:: sh

   python setup.py build_ext --inplace


CMake build options
-------------------

.. envvar:: MERLIN_CUDA

   Build C++ Merlin library with CUDA ``nvcc``.

   :Type: ``BOOL``
   :Value: ``ON``, ``OFF``
   :Default: ``ON``

.. envvar:: MERLIN_LIBKIND

   Specify the kind of compiled CUDA and C++ library.

   By default, compile dynamic library on Linux and static library on Windows.

   :Type: ``STRING``
   :Value: ``AUTO``, ``STATIC``, ``SHARED``
   :Default: ``AUTO``

.. envvar:: MERLIN_TEST

   Build test executable.

   :Type: ``BOOL``
   :Value: ``ON``, ``OFF``
   :Default: ``OFF``

.. |CMAKE_BUILD_TYPE| replace:: ``CMAKE_BUILD_TYPE``
.. _CMAKE_BUILD_TYPE: https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html

Build documentation
-------------------

The C++/CUDA documentation is retrieved by Doxygen and formatted in form of XML
files under ``docs/source/xml``. Later, ``Sphinx`` will read these files and
merge the C++/CUDA documentation with RST files and Python documentation,
forming a single result (can be HTML or PDF).

.. code-block:: sh

   cd docs
   doxygen Doxyfile
   make html

.. note::

   In order to build the documentation, the Python interface must have already
   been built or installed.
