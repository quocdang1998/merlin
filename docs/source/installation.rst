Installation
============

System requirements
-------------------

Prior to compiling the package, make sure that you have already installed these prerequisites:

-  C++ compiler: GNU ``g++>=13.2.0`` on **Linux** or Visual Studio 2022 ``cl>=19.41`` on **Windows**, with OpenMP
   enabled.

-  Cmake: ``cmake>=3.28.0``

-  Build-tool: GNU ``make`` on **Linux** or ``ninja`` on **Windows** (Ninja extenstion of MSVC).

-  CUDA ``nvcc>=12.6`` (optional, but required if GPU parallelization option is ``ON``).

.. _setup_script_build_dependancies:

To compile the Python interface by using the ``setup.py`` script, ensure the installation of the following modules:

-  |Pybind11|_

-  |Numpy|_

.. |Pybind11| replace:: ``Pybind11>=2.10``
.. _Pybind11: https://pypi.org/project/pybind11/
.. |Numpy| replace:: ``Numpy>1.17``
.. _Numpy: https://pypi.org/project/numpy/

.. code-block:: sh

   pip install -U pybind11 numpy

To compile the documentation, install the following packages:

-  |Doxygen|_

-  |Sphinx|_ + extensions (|sphinx_rtd_theme|_, |sphinx_design|_, |sphinxcontrib-bibtex|_, |breathe|_ and
   |sphinx_doxysummary|_)

.. code-block:: sh

   pip install -U sphinx_rtd_theme sphinx_design sphinxcontrib-bibtex breathe sphinx_doxysummary Sphinx

.. |Doxygen| replace:: ``Doxygen>=1.8.5``
.. _Doxygen: https://doxygen.nl/download.html
.. |Sphinx| replace:: ``Sphinx``
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. |sphinx_rtd_theme| replace:: ``sphinx_rtd_theme``
.. _sphinx_rtd_theme: https://sphinx-rtd-theme.readthedocs.io/en/stable/
.. |sphinx_design| replace:: ``sphinx_design``
.. _sphinx_design: https://sphinx-design.readthedocs.io/en/latest/
.. |sphinxcontrib-bibtex| replace:: ``sphinxcontrib-bibtex``
.. _sphinxcontrib-bibtex: https://sphinxcontrib-bibtex.readthedocs.io/en/latest/
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

On Windows, it is highly recommended to compile the library within the Visual Studio application. Inside the
application:

1. Configure CMake: **Project** :fa:`caret-right` **Configure merlin**

   .. image:: _img/installation_Configure.png
      :width: 100%

2. Build: **Build** :fa:`caret-right` **Build All**

   .. image:: _img/installation_Build.png
      :width: 100%

It is possible to compile the package from terminal (cmd or Powershell). However, users are responsible for ensuring the
correct configuration of environment variables before the compilation process, based on location and version of Visual
Studio installed on their machines (see also `Building on the command line
<https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#path_and_environment>`_ and
`Developper command prompt
<https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#developer_command_prompt_shortcuts>`_).

.. code-block:: powershell

   Launch-VsDevShell.ps1 -SkipAutomaticLocation -Arch amd64 # or vcvarsall x64
   cmake --preset=windows .
   cd build
   ninja

To customize the settings of the compilation of the library (e.g. compiling without CUDA), checkout
:ref:`installation:CMake build options`.

After the compilation step, executables, libraries and C++ header files can be installed using CMake command (note that
in the example below, current working directory is the one containing ``cmake_install.cmake``, i.e. ``build``):

.. code-block:: sh

   cmake --install . --prefix="/path/to/install/folder"
   # or cmake --install . --prefix='C:\path\to\install folder' on Windows

After the installation, environment variables must be set so compiler can find the package:

.. tab-set-code::

   .. code-block:: sh

      # suppose the package installed in "/path/to/install/folder"
      PATH=/path/to/install/folder/bin:$PATH
      CPATH=/path/to/install/folder/include:$PATH
      LD_LIBRARY_PATH=/path/to/install/folder/lib:$LD_LIBRARY_PATH

   .. code-block:: powershell

      # suppose the package installed in "C:\path\to\install folder"
      $env:PATH += ';C:\path\to\install folder\bin'
      $env:INCLUDE += ';C:\path\to\install folder\include'
      $env:LIB += ';C:\path\to\install folder\lib'

   .. code-block:: cmake

      find_package(OpenMP)      # required when compiling static Merlin library
      include(FindCUDAToolkit)  # required when compiling static Merlin library AND using CUDA
      # suppose the package installed in "/path/to/install/folder"
      find_package(merlin REQUIRED PATHS "/path/to/install/folder/lib/cmake")
      if(libmerlin_FOUND)
          message(STATUS "Found libmerlin cmake package")
      endif()
      # linking to custom executable
      add_executable(my_exe ${MY_SOURCE_LIST})
      set_property(TARGET my_exe PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)  # required!
      # these steps are required when using CUDA
      set_property(TARGET my_exe PROPERTY CUDA_SEPARABLE_COMPILATION ON)
      get_property(MERLIN_CUDA_ARCH TARGET merlin::libmerlin PROPERTY CUDA_ARCHITECTURES)
      set_property(TARGET my_exe PROPERTY CUDA_ARCHITECTURES ${MERLIN_CUDA_ARCH})
      target_link_libraries(executable PUBLIC merlin::libmerlin)


Python package
^^^^^^^^^^^^^^

The Python interface is a wrapper around the C++/CUDA library. Therefore, prior to compiling the Python interface,
verify that **the C++/CUDA interface has been successfully compiled**.

When compiling the Python module "inplace" (compiled extensions are copied to the source directory), :ref:`build
dependancies <setup_script_build_dependancies>` must be installed. Next, run the setup script with:

.. code-block:: sh

   python setup.py build_ext --inplace

The package can also be installed using ``pip``. If ``setuptools>=30``, the necessary build dependencies are
automatically installed during execution (in accordance with `PEP 517 <https://peps.python.org/pep-0517/>`_). Therefore
users are relieved from the obligation of manual pre-installation of the dependencies.

.. code-block:: sh

   pip install .


CMake build options
-------------------

Merlin offers 8 presets (compilation configurations). Preset names follow the following form:

.. code-block:: sh

   cmake --preset=[os][-cuda][-dev]

-  ``os`` (mandatory): Specify the operating system on which Merlin is compiled and utilized. The value can be either
   ``linux`` (using ``gcc`` and ``make``) or ``windows`` (using ``cl.exe`` and ``ninja``).

-  ``-cuda`` (optional): Compile using CUDA the GPU functionalities of Merlin. This option requires a CUDA Toolkit
   version of at least 12.6. The target GPUs are assumed to be attached to the CPU performing the compilation. Note that
   when this option is not enabled, **invoking functions reserved for GPUs will raise a runtime error**.

-  ``-dev`` (optional): Compile in debug mode, along with the unit test executables.

Examples:

.. code-block:: sh

   cmake --preset=windows-cuda      # Windows, release mode, CUDA enabled
   cmake --preset=linux-dev         # Linux,   debug mode,   CUDA disabled
   cmake --preset=windows-cuda-dev  # Windows, debug mode,   CUDA enabled

CMake variables for further customizing the compilation of C++/CUDA interface:

.. envvar:: MERLIN_CUDA

   Build C++ Merlin library with or without CUDA ``nvcc``.

   :Type: ``BOOL``
   :Value: ``ON``, ``OFF``
   :Default: ``OFF``

.. envvar:: MERLIN_DETECT_CUDA_ARCH

   Automatically detect the architectures of all GPUs connected to the CPU employed for compilation. Otherwise, the
   architectures fallback to the cache variable ``CMAKE_CUDA_ARCHITECTURES``.

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

.. envvar:: MERLIN_EXT

   Build C++ extensions to Merlin library.

   :Type: ``STRING``
   :Value: ``""``, ``"spgrid"``
   :Default: ``""``

Examples:

.. code-block:: sh

   cmake --preset=windows-cuda -DMERLIN_LIBKIND=SHARED
   cmake --preset=linux -DMERLIN_TEST=ON

Build documentation
-------------------

The C++/CUDA documentation is generated by Doxygen and organized as XML files in the directory ``docs/source/xml``.
Next, ``Sphinx`` conbines the C++/CUDA documentation and Python docstrings with RST files and creates a unified output,
which can be in the form of HTML or PDF.

.. code-block:: sh

   cd docs
   doxygen Doxyfile
   make html

.. note::

   In order to build the documentation, the Python interface must have already been built or installed, which requires
   the compilation of C++/CUDA library.
