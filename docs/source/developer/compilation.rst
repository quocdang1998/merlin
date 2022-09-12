Compilation scheme
==================

Hardware limitation prevents GPU threads from accessing binary code from disk at
runtime. **All CUDA** ``__device__`` **and** ``__host__ __device__`` **functions
must be either linked statically or inlined**. In some cases, the second option
is unavailable, such as when the content of the function is too complicated with
many nested loops, conditions and calls of non-inlined functions, or when it is
called by the Python interface (which uses the default C++ compiler instead of
CUDA compiler). The solution to this problem is to isolate the binary code of
all CUDA device functions in a library named ``libmerlincuda``
(see `CUDA_SEPARABLE_COMPILATION
<https://cmake.org/cmake/help/latest/prop_tgt/CUDA_SEPARABLE_COMPILATION.html>`_
). The source code of these functions are placed **in separate units suffixed
by** ``.rdc``. Files with this extension is **compiled as C++ source files if
the option** :envvar:`MERLIN_CUDA` **is** ``OFF`` **and as as CUDA source files
otherwise**.

The compilation scheme configured by Cmake depends on the kind of the library
chosen by user:

-  **Static library**: The process of compiling CUDA static library is
   similar a normal C++ library. Each source file is compiled into an object
   file, then archived in a library.

   .. image:: ../_img/static_comp.*
      :width: 175.99pt
      :height: 153pt
      :align: center
      :alt: Static library compilation

   To compile a binary or a dynamic library linking to the merlin library, one
   must perform a device linker before the usual linking step.

   .. tabs::

      .. code-tab:: sh

         g++ -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
         nvcc -shared -dlink -o device_code.o foo.o libmerlin.a
         g++ -o foo.exe foo.o device_code.o libmerlin.a

      .. code-tab:: powershell

         cl -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
         nvcc -shared -dlink -o device_code.obj foo.obj merlin.lib
         link /out:foo.exe foo.obj device_code.obj merlin.lib

      .. code-tab:: cmake

         add_executable(foo foo.cpp)  # or "add_executable(foo foo.cu)"
         set_property(TARGET foo PROPERTY CUDA_SEPARABLE_COMPILATION ON)
         target_link_libraries(foo libmerlin)


-  **Dynamic library**: Compilation process composes of 2 stages: creating a
   static library linking CUDA device code ``libmerlincuda``, and a dynamic
   library ``libmerlin`` containing the rest.

   .. image:: ../_img/dynamic_comp.*
      :width: 329.386pt
      :height: 271.912pt
      :align: center
      :alt: Dynamic library compilation

   To compile a binary or a dynamic library linking to the merlin library, one
   must perform a device linker before the regular linking process. Then, the
   linking process must link to both ``libmerlin`` and ``libmerlincuda``. Note
   that the linking order matters: ``libmerlin`` must be linked before
   ``libmerlincuda``.

   .. tabs::

      .. code-tab:: sh

         g++ -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
         nvcc -shared -dlink -o device_code.o foo.o libmerlin.a
         g++ -o foo.exe foo.o device_code.o libmerlin.so libmerlincuda.a

      .. code-tab:: powershell

         cl -c foo.cpp  # or "nvcc -c foo.cu" for CUDA applications
         nvcc -shared -dlink -o device_code.obj foo.obj merlin.lib
         link /out:foo.exe foo.obj device_code.obj merlin.lib merlincuda.lib

      .. code-tab:: cmake

         add_executable(foo foo.cpp)  # or "add_executable(foo foo.cu)"
         set_property(TARGET foo PROPERTY CUDA_SEPARABLE_COMPILATION ON)
         target_link_libraries(foo libmerlin libmerlincuda)

Although the compilation with Cmake supports both the compilation of static
library and dynamic library, it is recommended to use dynamic library on
Linux and static library on Windows for speed and simplicity.
