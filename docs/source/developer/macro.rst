Macro rules
===========

Some functions behaves differently depending on the build configuration, such as
working normally when CUDA option is enabled but yielding an error otherwise. In
order to control the compilation, macros are employed to suppress or redirect
the source code during the pre-processing step of the compiler. As a result, the
same file can be compiled without errors regardless of the compilation
configuration, thus avoiding multiple similar copies of the same source.

Here are some macros defined by this package during the compilation:

.. envvar:: __MERLIN_CUDA__

   :Condition: Defined when :envvar:`MERLIN_CUDA` is ``ON``.
   :Usage: Wrapping around definition of a host function depending on the CUDA
      option, or a class members visible only in CUDA configuration.

   .. code-block:: cpp

      // foo.hpp
      void foo(void);

      class Foo {
        public:
          int visible_;  // Visible regardless of CUDA option
          #ifdef __MERLIN_CUDA__
          int invisible_;  // Visible only when CUDA option enabled
          #endif  // __MERLIN_CUDA__
      };

      // foo.cpp
      #ifndef __MERLIN_CUDA__
      void foo(void) {
          FAILURE(cuda_compile_error,
                  "This function is defined only when CUDA option enabled.\n")
      }
      #endif  // __MERLIN_CUDA__

      // foo.cu
      void foo(void) {
          cuda_function();
      }

.. envvar:: __NVCC__

   :Condition: Defined when the compiler is CUDA ``nvcc``.
   :Usage: Wrapping around declaration or definition of inlined device
      functions in header, or template of device function in template.

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

   :Condition: Defined when the compiler is CUDA ``nvcc`` inside a
      ``__device__`` function.
   :Usage: Inside a ``__host__ __device__`` function definition with different
      implementation on CPU and GPU.

   .. code-block:: cpp

      // foo.rdc
      __host__ __device__ void foo(void) {
         #ifndef __CUDA_ARCH__
         cpu_function();
         #else
         gpu_function();
         #endif  // __CUDA_ARCH__
      }
