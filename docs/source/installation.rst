Installation
============

System requirements
-------------------

Before compiling the package, ensure that these prerequisites have already been
installed:

-  C++ compiler: GNU ``g++>=9.3.0`` on **Linux** or Visual Studio 2022
   ``cl.exe`` on **Windows**, with OpenMP enabled.

-  Cmake: ``cmake>=3.21.0`` (detect ``-std=c++17`` flag on CUDA).

-  Build-tool: GNU ``make`` on **Linux** or ``ninja`` on **Windows** (Ninja
   extenstion of MSVC).

-  CUDA ``nvcc>=11.4`` (optional, required if GPU parallelization option is
   ``ON``).

.. _setup_script_build_dependancies:

To compile the Python interface by using the ``setup`` script, install these
following modules:

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

.. code-block:: sh

      pip install -U numpy Jinja2 packaging
      pip install -U Cython --pre

.. note::

   By default, ``pip`` will install the newest release version of ``Cython``,
   which is ``0.29``, but the ``setup`` script requires the pre-release one.
   Thus, the argument ``--pre`` **must be passed** to the ``pip install``
   command.

To compile the documentation, install the following packages:

-  |Doxygen|_

-  |Sphinx|_ + extensions (|sphinx_rtd_theme|_, |sphinx_design|_,
   |breathe|_ and |sphinx_doxysummary|_)

   .. code-block:: sh

      pip install -U sphinx_rtd_theme sphinx_design breathe sphinx_doxysummary
      pip install -U Sphinx

.. |Doxygen| replace:: ``Doxygen>=1.8.5``
.. _Doxygen: https://doxygen.nl/download.html
.. |Sphinx| replace:: ``Sphinx``
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. |sphinx_rtd_theme| replace:: ``sphinx_rtd_theme``
.. _sphinx_rtd_theme: https://sphinx-rtd-theme.readthedocs.io/en/stable/
.. |sphinx_design| replace:: ``sphinx_design``
.. _sphinx_design: https://sphinx-design.readthedocs.io/en/latest/
.. |breathe| replace:: ``breathe``
.. _breathe: https://breathe.readthedocs.io/en/latest/
.. |sphinx_doxysummary| replace:: ``sphinx_doxysummary``
.. _sphinx_doxysummary: https://doxysummary.readthedocs.io/en/latest/


Compilation and Installation
----------------------------

C++ and CUDA
^^^^^^^^^^^^

On Linux, open a terminal and execute:

.. code-block:: sh

   cmake --preset=linux .
   cd build
   make -j

On Windows, because MSVC pre-defines environment variables, compilation inside
Visual Studio application is strongly recommended.

Inside the application:

1. Configure CMake: **Project** :fa:`caret-right` **Configure merlin**

   .. image:: _img/installation_Configure.png
      :width: 100%

2. Build: **Build** :fa:`caret-right` **Build All**

   .. image:: _img/installation_Build.png
      :width: 100%

It is possible to compile the package from the terminal (cmd or Powershell), but
user are responsible for assuring that enviroment variables are correctly set
before the compilation, depending on location and version of Visual Studio
installed on the machine (see also `Building on the command line
<https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#path_and_environment>`_
and `Developper command prompt
<https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#developer_command_prompt_shortcuts>`_).

.. code-block:: powershell

   cmake --preset=windows .
   cd build
   ninja

After the compilation step, executables, libraries and C++ header files can be
installed using CMake command (note that in the example below, current working
directory is the one containing ``cmake_install.cmake``, i.e. ``build``):

.. code-block:: sh

   cmake --install . --prefix="/path/to/install/folder/"
   # or cmake --install . --prefix="C:\\path\\to\\install\\folder\\" on Windows

In case of compilation of a dynamic library, the installation path must be
added to the environment variable ``LD_LIBRARY_PATH`` after the installation:

.. code-block:: sh

   export LD_LIBRARY_PATH=/path/to/install/folder:$LD_LIBRARY_PATH
   # or $env:LIB_PATH += "C:\\path\\to\\install\\folder\\" on Windows

To customize the settings of the compilation of the library (e.g. compiling
without CUDA), checkout :ref:`installation:CMake build options`.

Python package
^^^^^^^^^^^^^^

The Python interface is a module with C++ extensions calling classes and
functions from the C++/CUDA library. Thus, before compiling the Python
interface, **check that the C++/CUDA interface has been compiled**.

In case of compiling the Python module "inplace" (compiled extensions are copied
to the source directory), :ref:`build dependancies <setup_script_build_dependancies>`
must be installed. Next, run the setup script with options:

.. code-block:: sh

   python setup.py build_ext --inplace

The package can also be installed using ``pip``. By using ``setuptools>=30``,
build dependancies are installed automatically on the run (according to
the `PEP 517 <https://peps.python.org/pep-0517/>`_). User simply has to run:

.. code-block:: sh

   pip install .


CMake build options
-------------------

Options for customizing the compilation of C++/CUDA interface:

.. envvar:: MERLIN_CUDA

   Build C++ Merlin library with or without CUDA ``nvcc``.

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

   Build unit test executables.

   :Type: ``BOOL``
   :Value: ``ON``, ``OFF``
   :Default: ``OFF``

Build documentation
-------------------

The C++/CUDA documentation is retrieved by Doxygen and formatted in form of XML
files under ``docs/source/xml``. Next, ``Sphinx`` will read these files and
merge the C++/CUDA documentation with RST files and Python documentation,
forming a single result (can be HTML or PDF).

.. code-block:: sh

   cd docs
   doxygen Doxyfile
   make html

.. note::

   In order to build the documentation, the Python interface must have already
   been built or installed, which requires the compilation of C++/CUDA library.
