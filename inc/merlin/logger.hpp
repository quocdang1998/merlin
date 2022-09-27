// Copyright 2022 quocdang1998
#ifndef MERLIN_LOGGER_HPP_
#define MERLIN_LOGGER_HPP_

#include <cstdio>  // std::printf, std::vsnprintf
#include <cstdarg>  // std::va_list, va_start, va_end
#include <stdexcept>  // std::runtime_error
#include <type_traits>  // std::is_same
#include <system_error> // std::error_code
#include <filesystem>  // std::filesystem::filesystem_error

// Log MESSAGE, WARNING and FAILURE for CPU
// ----------------------------------------

// Macro expands to function name
#if defined(__GNUG__)
    #define __FUNCNAME__ __PRETTY_FUNCTION__
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define __FUNCNAME__ __FUNCSIG__
#endif

/** @brief Print message to the standard output.
 *  @details Example:
 *  @code {.cpp}
 *  MESSAGE("A message with a value %f.\n", 0.5);
 *  @endcode
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 */
#define MESSAGE(fmt, ...) std::printf("\033[1;34m[MESSAGE]\033[0m [%s] " fmt, __FUNCNAME__, ##__VA_ARGS__)
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
#define FAILURE(exception, fmt, ...) error_<exception>(__FUNCNAME__, fmt, ##__VA_ARGS__)

// print log for FAILURE + throw exception
template <class Exception = std::runtime_error>
void error_(const char * func_name, const char * fmt, ...) {
    // save formatted string to a buffer
    char buffer[1024];
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    // print exception message and throw an exception object
    std::fprintf(stderr, "\033[1;31m[FAILURE]\033[0m [%s] %s", func_name, buffer);
    if constexpr(std::is_same<Exception, std::filesystem::filesystem_error>::value) {
        throw std::filesystem::filesystem_error(const_cast<char *>(buffer), std::error_code(1, std::iostream_category()));
    } else {
        throw Exception(const_cast<char *>(buffer));
    }
}

/** @brief Exception class to be thrown when compiling without CUDA option.*/
class cuda_compile_error : public std::runtime_error {
  public:
    cuda_compile_error(const char * message) : std::runtime_error(message) {}
    const char * what() const noexcept {return std::runtime_error::what();}
};

/** @brief Exception class to be thrown when CUDA runtime error is encountered.*/
class cuda_runtime_error : public std::runtime_error {
  public:
    cuda_runtime_error(const char * message) : std::runtime_error(message) {}
    const char * what() const noexcept {return std::runtime_error::what();}
};

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
#endif  // __NVCC__

#endif  // MERLIN_LOGGER_HPP_
