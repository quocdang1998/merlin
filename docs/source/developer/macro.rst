Macro rules
===========

Some functions behaves differently depending on the build configuration, such as working normally when CUDA option is
enabled but yielding an error otherwise. In order to control the compilation, macros are employed to suppress or
redirect the source code during the pre-processing step of the compiler. As a result, the same file can be compiled
without errors regardless of the compilation configuration, thus avoiding multiple similar copies of the same source.

Here are some macros defined by this package during the compilation.

OS-dependant macros
^^^^^^^^^^^^^^^^^^^

.. envvar:: __MERLIN_WINDOWS__

   :Condition: Defined when the package is compiled on Windows and the compiler
      is MSVC.
   :Source: ``src/platform.hpp``.
   :Usage: Wrapping around included header files and function definitions for Windows.

   .. code-block:: cpp

      // include "windows.h" only if the OS is Windows
      #ifdef __MERLIN_WINDOWS__
          #include <windows.h>
      #endif // __MERLIN_WINDOWS__

      void foo(void) {
      #ifdef __MERLIN_WINDOWS__
          exclusive_implementation_on_windows();  // Executed only on Windows
      #endif // __MERLIN_WINDOWS__
      }

.. envvar:: __MERLIN_LINUX__

   :Condition: Defined when the package is compiled on Linux and the compiler
      is GNU ``g++``.
   :Source: ``src/platform.hpp``.
   :Usage: Wrapping around included header files and function definitions for Linux.

   .. code-block:: cpp

      // include "unistd.h" only if the OS is Linux
      #ifdef __MERLIN_LINUX__
          #include <unistd.h>
      #endif // __MERLIN_LINUX__

      void foo(void) {
      #ifdef __MERLIN_LINUX__
          exclusive_implementation_on_linux();  // Executed only on Linux
      #endif // __MERLIN_LINUX__
      }

CUDA-dependant macros
^^^^^^^^^^^^^^^^^^^^^

.. envvar:: __MERLIN_CUDA__

   :Condition: Defined when :envvar:`MERLIN_CUDA` is ``ON``.
   :Source: ``CMakeLists.txt``.
   :Usage: Wrapping around definition of a host function in ``cpp`` source that depends on the CUDA option, or a class
      members visible only in CUDA configuration.

   .. code-block:: cpp

      // foo.hpp
      void foo(void);

      class Foo {
        public:
          int visible_;  // Visible regardless of CUDA option
      #ifdef __MERLIN_CUDA__
          int gpu_visible_;  // Visible only when CUDA option enabled
      #endif  // __MERLIN_CUDA__
      };

      // foo.cpp (definition when CUDA option disabled)
      #ifndef __MERLIN_CUDA__
      void foo(void) {
          Fatal<cuda_compile_error>("Function unavailable without CUDA.\n");
      }
      #endif  // __MERLIN_CUDA__

      // foo.cu (definition when CUDA option enabled)
      void foo(void) {
          cuda_function();
      }

.. envvar:: __NVCC__

   :Condition: Defined when the compiler is CUDA ``nvcc``.
   :Source: Native with ``nvcc`` compiler.
   :Usage: Wrapping around declaration or definition of inlined device functions in header, or template of device
      function in template.

   .. code-block:: cpp

      // foo.hpp
      #ifdef __NVCC__
      __device__ void foo(void);

      __device__ inline void foo_inline(void) {
          do_sth();
      }
      #endif  // __NVCC__

      // foo.tpp
      #ifdef __NVCC__
      template <typename T>
      __device__ T add(T a, T b) {
          return a+b;
      }
      #endif  // __NVCC__

.. envvar:: __CUDA_ARCH__

   :Condition: Defined when the compiler is CUDA ``nvcc`` inside a ``__device__`` function.
   :Source: Native with ``nvcc`` compiler.
   :Usage: Inside a ``__host__ __device__`` function definition with different implementation on CPU and GPU.

   .. code-block:: cpp

      // foo.rdc
      __host__ __device__ void foo(void) {
      #ifndef __CUDA_ARCH__
          cpu_function();
      #else
          gpu_function();
      #endif  // __CUDA_ARCH__
      }

Library kind dependant macros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: __MERLIN_BUILT_AS_STATIC__

   :Condition: Defined at compilation of static library.
   :Source: ``CMakeLists.txt``.

.. envvar:: __LIBMERLINCUDA__

   :Condition: Defined at compilation of ``libmerlincuda``.
   :Source: ``CMakeLists.txt``.

Export macros for dynamic library on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: MERLIN_EXPORTS

   :Condition: Defined at compilation of dynamic library ``libmerlin`` on
      Windows.
   :Source: ``exports.hpp``.
   :Usage: Append before functions and classes that are linked dynamically with
      the dynamic library ``merlin.dll``.
   :Note: This macro with expands to empty when compiling on Linux, or when
      compiling static library (:envvar:`MERLIN_LIBKIND` is ``STATIC``).

.. envvar:: MERLINENV_EXPORTS

   :Condition: Defined at compilation of dynamic library ``libmerlinenv`` on
      Windows.
   :Source: ``exports.hpp``.
   :Usage: Append before functions and classes that are linked dynamically with
      the dynamic library ``merlinenv.dll``.
   :Note: Similar to :envvar:`MERLIN_EXPORTS`, this macro with expands to empty
      when compiling on Linux.
