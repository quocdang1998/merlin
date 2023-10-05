// Copyright 2022 quocdang1998
#ifndef MERLIN_LOGGER_HPP_
#define MERLIN_LOGGER_HPP_

#include <cstdarg>       // std::va_list, va_start, va_end
#include <cstdio>        // std::printf, std::vsnprintf
#include <filesystem>    // std::filesystem::filesystem_error
#include <stdexcept>     // std::runtime_error
#include <string>        // std::string
#include <system_error>  // std::error_code
#include <type_traits>   // std::is_same

#ifdef __NVCC__
    #include "cuda.h"  // CUresult, cuGetErrorName
#endif                 // __NVCC__

#include "merlin/exports.hpp"   // MERLINSHARED_EXPORTS
#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__

#if defined(__MERLIN_WINDOWS__)
    #define __FUNCNAME__ __FUNCSIG__
#elif defined(__MERLIN_LINUX__)
    #define __FUNCNAME__ __PRETTY_FUNCTION__
#endif

// Stack tracing
// -------------

namespace merlin {
/** @brief Print the stacktrace at the crash moment.*/
MERLINSHARED_EXPORTS void print_stacktrace(int skip = 1);
}  // namespace merlin

// Log MESSAGE, WARNING and FAILURE for CPU
// ----------------------------------------

/** @brief Print message to the standard output.
 *  @details Example:
 *  @code {.cpp}
 *  MESSAGE("A message with a value %f.\n", 0.5);
 *  @endcode
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 */
#define MESSAGE(fmt, ...) std::fprintf(stdout, "\033[1;34m[MESSAGE]\033[0m [%s] " fmt, __FUNCNAME__, ##__VA_ARGS__)
/** @brief Print warning to the standard error.
 *  @details Example:
 *  @code {.cpp}
 *  int x = 2;
 *  WARNING("A warning with a value %d.\n", x);
 *  @endcode
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 */
#define WARNING(fmt, ...) std::fprintf(stderr, "\033[1;33m[WARNING]\033[0m [%s] " fmt, __FUNCNAME__, ##__VA_ARGS__)
/** @brief Print error message and throw an instance of type exception.
 *  @details Example:
 *  @code {.cpp}
 *  FAILURE(std::runtime_error, "A runtime error message.\n");
 *  @endcode
 *  @note The use of this macro is recommended over C++ exception throwing, because some compiler/debugger may be
 *  unable to display automatically the error message.
 *  @param exception Name of the exception class (like ``std::runtime_error``, ``std::invalid_argument``, etc).
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 */
#define FAILURE(exception, fmt, ...) ::merlin::__throw_error<exception>(__FUNCNAME__, fmt, ##__VA_ARGS__)
/** @brief Perform a check only in debug mode.
 *  @details Example:
 *  @code {.cpp}
 *  CASSERT(i < j, FAILURE, std::runtime_error, "A runtime error message.\n");
 *  @endcode
 */
#if defined(__MERLIN_DEBUG__)
    #define CASSERT(condition, ERROR_MACRO, exception, fmt, ...)                                                       \
        if (condition)                                                                                                 \
            ERROR_MACRO(exception, fmt, ##__VA_ARGS__)
#else
    #define CASSERT(condition, ERROR_MACRO, exception, fmt, ...)
#endif  // __MERLIN_DEBUG__

namespace merlin {

// print log for FAILURE + throw exception
template <class Exception = std::runtime_error>
void __throw_error(const char * func_name, const char * fmt, ...) {
    // save formatted string to a buffer
    char buffer[1024];
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    // print exception message and throw an exception object
    std::fprintf(stderr, "\033[1;31m[FAILURE]\033[0m [%s] %s", func_name, buffer);
#if defined(__MERLIN_DEBUG__)
    print_stacktrace(2);  // print_stacktrace function
#endif                    // __MERLIN_DEBUG__
    if constexpr (std::is_same<Exception, std::filesystem::filesystem_error>::value) {
        throw std::filesystem::filesystem_error(const_cast<char *>(buffer),
                                                std::error_code(1, std::iostream_category()));
    } else {
        throw Exception(const_cast<char *>(buffer));
    }
}

/** @brief Exception class to be thrown when compiling without CUDA option.*/
class cuda_compile_error : public std::runtime_error {
  public:
    cuda_compile_error(const char * message) : std::runtime_error(message) {}
    const char * what() const noexcept { return std::runtime_error::what(); }
};

/** @brief Exception class to be thrown when CUDA runtime error is encountered.*/
class cuda_runtime_error : public std::runtime_error {
  public:
    cuda_runtime_error(const char * message) : std::runtime_error(message) {}
    const char * what() const noexcept { return std::runtime_error::what(); }
};

/** @brief Exception class to be thrown when a feature that is not yet implemented.*/
class not_implemented_error : public std::runtime_error {
  public:
    not_implemented_error(const char * message) : std::runtime_error(message) {}
    const char * what() const noexcept { return std::runtime_error::what(); }
};

}  // namespace merlin

// Get error message from Windows and Unix
// ---------------------------------------

namespace merlin {
#if defined(__MERLIN_WINDOWS__)
// Get error from Windows API
MERLINSHARED_EXPORTS std::string throw_windows_last_error(unsigned long int last_error);
#elif defined(__MERLIN_LINUX__)
// Get error from Linux
MERLINSHARED_EXPORTS std::string throw_linux_last_error(void);
#endif
}  // namespace merlin

// Log CUDAOUT for GPU
// -------------------

#ifdef __NVCC__
    /** @brief Print message to the standard output (for usage inside a GPU function).
     *  @details Example:
     *  @code {.cu}
     *  __global__ void dummy_gpu_function {
     *      CUDAOUT("Hello World from thread %d block %d.\n", threadIdx.x, blockIdx.x);
     *  }
     *  @endcode
     *  @note This macro is only available when option ``MERLIN_CUDA`` is ``ON``.
     *  @param fmt Formatted string (same syntax as ``std::printf``).
     */
    #define CUDAOUT(fmt, ...) std::printf("\033[1;36m[CUDAOUT]\033[0m [%s] " fmt, __FUNCNAME__, ##__VA_ARGS__)

    /** @brief Print error message (for usage inside a GPU function) and terminate GPU kernel.
     *  @details Example:
     *  @code {.cu}
     *  __global__ void errorred_gpu_function {
     *      CUDAERR("Error from thread %d block %d.\n", threadIdx.x, blockIdx.x);
     *  }
     *  @endcode
     *  @note This macro is only available when option ``MERLIN_CUDA`` is ``ON``.
     *  @param fmt Formatted string (same syntax as ``std::printf``).
     */
    #define CUDAERR(fmt, ...)                                                                                          \
        std::printf("\033[1;35m[CUDAERR]\033[0m [%s] " fmt, __FUNCNAME__, ##__VA_ARGS__);                              \
        asm("trap;")
#endif  // __NVCC__

// Log CUHDERR for host-device error
// ---------------------------------

#ifdef __CUDA_ARCH__
    #define CUHDERR(exception, fmt, ...) CUDAERR(fmt, ##__VA_ARGS__)
#else
    /** @brief Print error message and terminate CUDA host-device function.
     *  @details Example:
     *  @code {.cu}
     *  __host__ __device__ void errorred_function {
     *      CUDAERR("Error.\n");
     *  }
     *  @endcode
     *  @param exception Name of the exception class (like ``std::runtime_error``, ``std::invalid_argument``, etc) to be
     *  thrown in case of ``__host__`` function.
     *  @param fmt Formatted string (same syntax as ``std::printf``).
     */
    #define CUHDERR(exception, fmt, ...) FAILURE(exception, fmt, ##__VA_ARGS__)
#endif  // __CUDA_ARCH__

#endif  // MERLIN_LOGGER_HPP_
