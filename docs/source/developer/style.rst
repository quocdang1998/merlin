Code styling
============

ClangFormat
-----------

C++/CUDA sources formatter is ``clang-format`` (see `ClangFormat <https://clang.llvm.org/docs/ClangFormat.html>`_).
Style configuration is detailed in ``.clang-format``.

.. code-block:: sh

   find src/merlin -name *.[hct]pp -or -name *.rdc -or -name *.glb | xargs clang-format -n

Besides the standard style rules, some additional rules are also applied:

*  Source files always starts with the copyright comment. Developer must enter the year followed by her/his name (see
   the example below).

*  Developer must list all classes/functions/templates included from headers. This allows the safe removal of the header
   file when modifications are made to the source.

*  Each include must be followed by classe/function names called from included header, listed in a line comment. There
   are 4 categories of include files, separated by blank lines. Order of include is:

   #. Standard library header files, like ``algorithm``, ``cstdlib``, ``cstring``, ``cstdint``, ``iostream``,
      ``string``.

   #. CUDA driver library's header (``cuda.h``) and OpenMP's header (``omp.h``).

   #. External libraries' header files, like ``gsl/gsl_foo.h``, ``fmt/core.h``.

   #. Project's header files, like ``merlin/utils.hpp``, ``merlin/logger.hpp``.

   .. code-block:: c++
      :linenos:

      // Copyright 2022 quocdang1998

      #include <cstring>           // std::memcpy
      #include <initializer_list>  // std::initializer_list
      #include <vector>            // std::vector

      #include <cuda.h>  // ::cuDeviceGetName
      #include <omp.h>   // ::omp_get_thread_num

      #include "merlin/array/array.hpp"  // merlin::array::Array
      #include "merlin/logger.hpp"       // Message, Fatal

*  Indent is 4 spaces by default. Elements in between header guards, namespaces and macros keep the same indent level as
   its parents. Class encapsulation indicators (``public``, ``protected`` and ``private``) are indented by 2 spaces.

   .. code-block:: c++
      :linenos:

      // foo.hpp
      #ifndef FOO_HPP_
      #define FOO_HPP_

      namespace merlin {

      #ifdef __NVCC__
      void foo(void);
      #endif  // __NVCC__

      class Foo {
        public:
          void public_method(void);

        protected:
          void protected_method(void);

        private:
          void private_method(void);
      };

      }  // namespace merlin

      #endif  // FOO_HPP_

*  Use C++11 pragma operator ``_Pragma("...")`` instead of the pre-processing form ``#pragma ...``. This allows
   ``clang-format`` to indent pragma directives properly as a C++ command, not as a preprocessor.

   .. code-block:: c++
      :linenos:

      // not recommended
      #pragma omp parallel num_threads(8)
      {
          // parallel region
          #pragma omp critical
          {
              // critical region
          }
          #pragma omp barrier
      }

      // use this instead
      _Pragma("omp parallel num_threads(8)") {
          // parallel region
          _Pragma("omp critical") {
              // critical region
          }
          _Pragma("omp barrier");
      }

.. note::

   All the rules listed above are not applied on C++ extensions of the Python interface.

Ruff
----

Python sources follow the configuration by ``ruff`` (see `RuffFormatter <https://docs.astral.sh/ruff/formatter/>`_).
This upon the guidelines outlined in PEP-8 (see `Style Guide for Python Code <https://peps.python.org/pep-0008/>`_), and
allows line lengths of up to 120 characters.

.. code-block:: sh

   ruff check setup_cfg/
   ruff format setup_cfg/

cmakelang
---------

CMake sources are formatted using ``cmake-format`` (see `cmakelang <https://cmake-format.readthedocs.io/en/latest/>`_).
The configuration for the formatter is detailed in ``.cmake-format.yaml``.

.. code-block:: sh

   cmake-format CMakeLists.txt
   cmake-format cmake/*.cmake
