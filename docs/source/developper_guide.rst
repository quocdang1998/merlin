Developer Guide
===============

C++ sources
-----------


Source files of C++ API are placed in folders ``inc/merlin`` and ``src``:

.. literalinclude:: _img/source_tree.txt
   :language: sh

The Cmake configuration file supports both the compilation of static library
(machine code is copied into the final executable) and dynamic library (only
the dynamic link appears in the executable) through the option
:envvar:`MERLIN_LIBKIND`. However, it is recommended to use dynamic library on
Linux and static library on Windows to simplify the development effort. If
the kind of the library is not the recommended mode, it is users'
responsibility to fix the issue.

The linking to dynamic library is specifically complicated, because **CUDA device
code can only be linked statically**. Hence compilation of a dynamic library with
CUDA option result in 2 files: a static library named libmerlincuda containing only
device codes, and a dynamic library containing the rest. In order to avoid the
duplicated definition of non-inline device functions, 2 macros has been defined:
:envvar:`__LIBMERLIN_STATIC__` and :envvar:`__MERLIN_FORCE_STATIC__` to instruct
the compiler to compile only device code and ignore host code.

Macro rules
^^^^^^^^^^^

Since compilers pre-process source files before the compilation, macro definition
and conditional compilation can guide the compilers which part of the source
files must be included and which part must be ignored. Hence, the same file can
be used and compiled without error regardless of the compilation configuration,
thus avoiding having multiple similar copies of the source.

Here are 4 macros defined by project Merlin to help redirecting the source
code:

.. envvar:: __MERLIN_CUDA__

   :Definition: Defined when Cmake is configured with CUDA option enabled.
   :Description: Used in function source code, when the definition depends on
      the CUDA option.
   :Usage: Inside a **host function calling other functions from CUDA Runtime
      Library**, or **class members needed only in CUDA configuration**.

.. envvar:: __MERLIN_FORCE_STATIC__

   :Definition: Defined when compiled library is static.
   :Description: Used in function source code, along with
      :envvar:`__LIBMERLIN_STATIC__`
   :Usage: Around **non-inline host-device function and device function
      definitions**.

.. envvar:: __LIBMERLIN_STATIC__

   :Definition: Defined when compiling the static library ``libmerlincuda``.
   :Description: Used in function source code, to filter function definitions
      included in the static library ``libmerlincuda``.
   :Usage: Around **non-inline host-device function and device function
      definitions**.

.. envvar:: __NVCC__

   :Definition: Defined by CUDA compiler.
   :Description: Used in header files included by both C++ sources and CUDA
      sources.
   :Usage: Wrapping around a **device function declaration**, or **definition
      of template device function** to avoid the C++ compiler miss-treatment of
      CUDA decorators.

.. envvar:: __CUDA_ARCH__

   :Definition: Defined by CUDA compiler inside a device function.
   :Description: Used inside a function definition when its behavior is different
      on CPU and GPU.
   :Usage: Inside a host-device function.


The code style of C++ code is Google's style guide, checked by ``cpplint``.

.. code-block:: sh

   cpplint --root=inc inc/merlin/*.hpp
   cpplint src/*.hpp

General rule for source files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  Header files always starts with the copyright comment.

   .. code-block:: c++
      :linenos:

      // Copyright 2022 quocdang1998

*  Each include must be followed by classes/functions called from included
   library, listed in a line comment. There are 4 group of include files, each
   group must be put between 2 blank lines. Order of include group is:

   #. C header files, like ``cstdio``, ``cstdlib``, ``cstring``, ``cstdint``.

   #. C++ header files, like ``iostream``, ``string``, ``tuple``, ``vector``.

   #. External libraries' header files, like ``gsl/gsl_foo.h``, ``fmt/core.h``.

   #. Project's header files, like ``merlin/utils.hpp``, ``merlin/logger.hpp``.

   .. code-block:: c++
      :linenos:

      // Copyright 2022 quocdang1998

      #include <cstring>  // std::memcpy

      #include <initializer_list>  // std::inializer_list
      #include <vector>  // std::vector

      #include "merlin/logger.hpp"  // MESSAGE, FAILURE

*  There are at least 2 spaces between a line comment and the code.

*  All declaration and definition must be enclosed in namespace. The use of
   ``using namspace`` is highly unrecommended.

   .. code-block:: c++
      :linenos:

      namespace merlin {

      // your code here

      }  // namespace merlin

*  Yes
