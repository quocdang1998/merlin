Compilation scheme
==================

Shared library
--------------

In the circumtances of the Python interface, C++ extern variables and class
static attributes need to be shared between all compiled binary extensions.
Different Python binary modules (which are dynamic linked libraries) must refer
to the same object when they call the same symbolic name. Thus, these variables
and attributes must be isolated inside a shared library.

.. image:: ../_img/shared_comp.*
   :align: center
   :alt: Shared library compilation

The library ``libmerlinshared`` is always compiled regardless of the kind of the
chosen library or the CUDA option.

Static library compilation
--------------------------

In case of static library, the process of compiling CUDA static library is
similar to a normal C++ library. Each source file is compiled into an object
file, then archived in a library.

   .. image:: ../_img/static_comp.*
      :align: center
      :alt: Static library compilation

To compile a binary or a dynamic library linking to the merlin library, if CUDA
option is enabled, one must perform a device linker before the usual linking
step.

.. tab-set-code::

   .. code-block:: sh

      g++ -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
      nvcc -shared -dlink -o device_code.o foo.o libmerlin.a
      g++ -o foo.exe foo.o device_code.o libmerlin.a libmerlinshared.so

   .. code-block:: powershell

      cl -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
      nvcc -shared -dlink -o device_code.obj foo.obj merlin.lib
      link /out:foo.exe foo.obj device_code.obj merlin.lib merlinshared.lib

   .. code-block:: cmake

      add_executable(foo foo.cpp)  # or "add_executable(foo foo.cu)"
      set_property(TARGET foo PROPERTY CUDA_SEPARABLE_COMPILATION ON)
      target_link_libraries(foo libmerlin)

CUDA library
------------

Compiling a dynamic library is trickier with CUDA option enabled, for the reason
that hardware limitation prevents GPU threads from accessing binary code from
disk at runtime. Thus, all CUDA ``__device__``, ``__host__ __device__`` and
``__global__`` functions must be either linked statically or inlined.

In some cases, the second option is unavailable, such as when the content of the
function is too complicated with many nested loops, conditions and calls of
non-inlined functions, or when it is called by the Python interface (which uses
the default C++ compiler instead of CUDA compiler). The solution to this problem
is to isolate the binary code of all CUDA host-device functions and global
functions in static libraries (see `CUDA_SEPARABLE_COMPILATION
<https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html>`_
).

The source code of these librairies are suffixed by ``.rdc`` and ``.glb`` 
respectively to distinguish them from ``*.cu`` CUDA sources calling functions
from the CUDA Runtime and CUDA Driver library. Files with suffix ``rdc`` is
compiled as C++ source files if the option :envvar:`MERLIN_CUDA` is ``OFF`` and
as as CUDA source files otherwise. Files with suffix ``glb`` is compiled as CUDA
source files only if the option :envvar:`MERLIN_CUDA` is enabled.

.. image:: ../_img/cuda_comp.*
   :align: center
   :alt: CUDA static library compilation

Static librairies ``libmerlincuda`` and ``libmerlinglobal`` are only compiled if
CUDA option is enabled and dynamic library compilation option is chosen. The
first library is meant to be used in the device linking step, while the role of
the second is to be linked against the final binary, alongside ``libmerlin``.

Dynamic library compilation
---------------------------

Compilation process of dynamic library composes of 2 stages: creating static
libraries linking CUDA device code ``libmerlincuda`` and CUDA global functions
``libmerlinglobal``, and the main dynamic library ``libmerlin`` containing the
rest (C++ and CUDA runtime code).

.. image:: ../_img/dynamic_comp.*
   :align: center
   :alt: Dynamic library compilation

To compile a binary or a dynamic library linking to the merlin library, if CUDA
option is enabled, one must perform a device linker against ``libmerlinglobal``
and ``libmerlincuda`` before the regular linking process. Next, the linking
process must link to both ``libmerlin``, ``libmerlinglobal`` and
``libmerlincuda``. Note that the linking order matters: ``libmerlin`` and
``libmerlinglobal`` must be passed to the linker before ``libmerlincuda``.

.. tab-set-code::

   .. code-block:: sh

      g++ -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
      nvcc -shared -dlink -o device_code.o foo.o libmerlinglobal.a libmerlin.a
      g++ -o foo.exe foo.o device_code.o libmerlin.so libmerlinglobal.a libmerlincuda.a libmerlinshared.so

   .. code-block:: powershell

      cl -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
      nvcc -shared -dlink -o device_code.obj foo.obj merlinglobal.lib merlincuda.lib
      link /out:foo.exe foo.obj device_code.obj merlin.lib merlinglobal.lib merlincuda.lib merlinshared.lib

   .. code-block:: cmake

      add_executable(foo foo.cpp)  # or "add_executable(foo foo.cu)"
      set_property(TARGET foo PROPERTY CUDA_SEPARABLE_COMPILATION ON)
      target_link_libraries(foo libmerlin libmerlinglobal)

Although the compilation with Cmake supports both the compilation of static
library and dynamic library, it is recommended to use dynamic library on
Linux and static library on Windows for speed and simplicity.
