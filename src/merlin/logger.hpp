// Copyright 2022 quocdang1998
#ifndef MERLIN_LOGGER_HPP_
#define MERLIN_LOGGER_HPP_

#include <cinttypes>        // PRIdLEAST32
#include <cstdio>           // std::snprintf
#include <filesystem>       // std::filesystem::filesystem_error
#include <iostream>         // std::cout, std::clog
#include <source_location>  // std::source_location
#include <stdexcept>        // std::runtime_error
#include <syncstream>       // std::osyncstream
#include <utility>          // std::forward

#include "merlin/config.hpp"    // merlin::printf_buffer, __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"   // MERLIN_EXPORTS

// Stack tracing
// -------------

namespace merlin {
/** @brief Print the stacktrace at the crash moment.*/
MERLIN_EXPORTS void print_stacktrace(int skip = 1);


// Get error message from Windows and Unix
// ---------------------------------------

/** @brief Get error message from Windows or Linux.*/
MERLIN_EXPORTS std::string throw_sys_last_error(unsigned long int last_error = static_cast<unsigned long int>(-1));

// Custom exceptions
// -----------------

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

// Log Message, Warning and Fatal for CPU
// --------------------------------------

/** @brief Wrapper around format string and current source location.*/
struct FmtString {
  public:
    consteval FmtString(const char * data, std::source_location location = std::source_location::current()) :
    str(data), loc(location) {}
    const char * str;
    std::source_location loc;
};

/** @brief Get length of a string.*/
constexpr int len(const char * str) {
    int length = 0;
    while (*(str++) != '\0') {
        ++length;
    }
    return length;
}

/** @brief Concatenate 2 strings.*/
constexpr void cat_str(char * dest_end, const char * src) {
    do {
        *(dest_end++) = *src;
    } while (*(src++) != '\0');
}

/** @brief Get filename from absolute path.*/
constexpr const char * get_fname(const char * abs_path) {
    const char * fname = abs_path;
    for (; *abs_path != '\0'; abs_path++) {
        if ((*abs_path == '/') || (*abs_path == '\\')) {
            fname = abs_path + 1;
        }
    }
    return fname;
}

/** @brief Print message to the standard output.
 *  @details Example:
 *  @code {.cpp}
 *  // use case 1: single output
 *  Message("A message with a value ", 0.5, ".\n");
 *  // use case 2: multiple output
 *  Message m("A printf ");
 *  m << "that takes " << "extra arguments. ";
 *  m << "And this message will not be mixed even in multi-threading.";
 *  @endcode
 */
struct Message {
  public:
    /** @brief Constructor list of objects.*/
    template <typename... Args>
    Message(FmtString fmt, Args &&... args) {
        char srcinfo_buffer[printf_buffer];
        std::snprintf(srcinfo_buffer, printf_buffer, "message [%s] in %s l.%" PRIdLEAST32 " : ", fmt.loc.function_name(),
                     get_fname(fmt.loc.file_name()), fmt.loc.line());
        char log_buffer[printf_buffer];
        if constexpr (sizeof...(args) > 0) {
            std::snprintf(log_buffer, printf_buffer, fmt.str, std::forward<Args>(args)...);
        } else {
            std::snprintf(log_buffer, printf_buffer, "%s", fmt.str);
        }
        this->stream << srcinfo_buffer << log_buffer;
    }
    /** @brief Stream operator.*/
    template <typename T>
    Message & operator<<(T && obj) {
        this->stream << obj;
        return *this;
    }
    /** @brief Force print out content to the output stream.*/
    void emit(void) { this->stream.emit(); }

  private:
    std::osyncstream stream{std::cout};
};

/** @brief Print warning to the standard error.
 *  @details Example:
 *  @code {.cpp}
 *  // use case 1: single output
 *  Warning("A warning with a value ", 0.5, ".\n");
 *  // use case 2: multiple output
 *  Warning m("A warning ");
 *  m << "that takes " << "extra arguments. ";
 *  m << "And this warning will not be mixed even in multi-threading.";
 *  @endcode
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 */
struct Warning {
  public:
    /** @brief Constructor list of objects.*/
    template <typename... Args>
    Warning(FmtString fmt, Args &&... args) {
        char srcinfo_buffer[printf_buffer];
        std::snprintf(srcinfo_buffer, printf_buffer, "warning [%s] in %s l.%" PRIdLEAST32 " : ", fmt.loc.function_name(),
                     get_fname(fmt.loc.file_name()), fmt.loc.line());
        char log_buffer[printf_buffer];
        if constexpr (sizeof...(args) > 0) {
            std::snprintf(log_buffer, printf_buffer, fmt.str, std::forward<Args>(args)...);
        } else {
            std::snprintf(log_buffer, printf_buffer, "%s", fmt.str);
        }
        this->stream << srcinfo_buffer << log_buffer;
    }
    /** @brief Stream operator.*/
    template <typename T>
    Warning & operator<<(T && obj) {
        this->stream << obj;
        return *this;
    }
    /** @brief Force print out content to the output stream.*/
    void emit(void) { this->stream.emit(); }

  private:
    std::osyncstream stream{std::clog};
};

/** @brief Message that is printed only in debug mode.
 *  @details Example:
 *  @code {.cpp}
 *  DebugLog("A message with a value %f.\n", 0.5);
 *  @endcode
 */
template <typename... Args>
struct DebugLog {
  public:
    /** @brief Constructor list of objects.*/
    DebugLog(FmtString fmt, Args &&... args) {
#if defined(__MERLIN_DEBUG__)
        char srcinfo_buffer[printf_buffer];
        std::snprintf(srcinfo_buffer, printf_buffer, "debug [%s] in %s l.%" PRIdLEAST32 " : ", fmt.loc.function_name(),
                     get_fname(fmt.loc.file_name()), fmt.loc.line());
        char log_buffer[printf_buffer];
        if constexpr (sizeof...(args) > 0) {
            std::snprintf(log_buffer, printf_buffer, fmt.str, std::forward<Args>(args)...);
        } else {
            std::snprintf(log_buffer, printf_buffer, "%s", fmt.str);
        }
        std::fprintf(stdout, "%s%s", srcinfo_buffer, log_buffer);
        print_stacktrace(2);
#endif  // __MERLIN_DEBUG__
    }
};

/** @brief Print error message and throw an instance of type exception.
 *  @details Example:
 *  @code {.cpp}
 *  Fatal<std::runtime_error>(std::runtime_error, "A runtime error message.\n");
 *  @endcode
 *  @note The use of this macro is recommended over C++ exception throwing, because some compiler/debugger may be
 *  unable to display automatically the error message.
 *  @tparam exception Name of the exception class (like ``std::runtime_error``, ``std::invalid_argument``, etc).
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 *  @param args Other arguments.
 */
template <class Exception>
struct Fatal {
  public:
    /** @brief Constructor from formatted string.*/
    template <typename... Args>
    Fatal(FmtString fmt, Args &&... args) {
        char srcinfo_buffer[printf_buffer];
        std::snprintf(srcinfo_buffer, printf_buffer, "fatal [%s] in %s l.%" PRIdLEAST32 " : ", fmt.loc.function_name(),
                     get_fname(fmt.loc.file_name()), fmt.loc.line());
        char error_buffer[printf_buffer];
        if constexpr (sizeof...(args) > 0) {
            std::snprintf(error_buffer, printf_buffer, fmt.str, std::forward<Args>(args)...);
        } else {
            std::snprintf(error_buffer, printf_buffer, "%s", fmt.str);
        }
        std::fprintf(stderr, "%s%s", srcinfo_buffer, error_buffer);
#if defined(__MERLIN_DEBUG__)
        print_stacktrace(2);
#endif  // __MERLIN_DEBUG__
        if constexpr (std::is_same<Exception, std::filesystem::filesystem_error>::value) {
            throw std::filesystem::filesystem_error(error_buffer, std::error_code(1, std::iostream_category()));
        } else {
            throw Exception(error_buffer);
        }
    }
};

// Log CudaOut, CudaDeviceError and CudaHostDevError for GPU
// ---------------------------------------------------------

#if defined(__NVCC__)

/** @brief Print message to the standard output (for usage inside a GPU function).
 *  @details Example:
 *  @code {.cu}
 *  __global__ void dummy_gpu_function {
 *      CudaOut("Hello World from thread %d block %d.\n", threadIdx.x, blockIdx.x);
 *  }
 *  @endcode
 *  @note This macro is only available when option ``MERLIN_CUDA`` is ``ON``.
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 *  @param args Other arguments.
 */
struct CudaOut {
    /** @brief Constructor from formatted string.*/
    template <typename... Args>
    __cudevice__ CudaOut(FmtString fmt, Args &&... args) {
        char buffer[printf_buffer] = "cudaout [%s] in %s l.%u : ";
        cat_str(buffer + len(buffer), fmt.str);
        std::printf(buffer, fmt.loc.function_name(), get_fname(fmt.loc.file_name()), unsigned(fmt.loc.line()),
                    std::forward<Args>(args)...);
    }
};

/** @brief Print error message (for usage inside a GPU function) and terminate GPU kernel.
 *  @details Example:
 *  @code {.cu}
 *  __global__ void errorred_gpu_function {
 *      DeviceError("Error from thread %d block %d.\n", threadIdx.x, blockIdx.x);
 *  }
 *  @endcode
 *  @note This macro is only available when option ``MERLIN_CUDA`` is ``ON``.
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 *  @param args Other arguments.
 */
struct DeviceError {
    /** @brief Constructor from formatted string.*/
    template <typename... Args>
    __cudevice__ DeviceError(FmtString fmt, Args &&... args) {
        char buffer[printf_buffer] = "cudaerror [%s] in %s l.%u : ";
        cat_str(buffer + len(buffer), fmt.str);
        std::printf(buffer, fmt.loc.function_name(), get_fname(fmt.loc.file_name()), unsigned(fmt.loc.line()),
                    std::forward<Args>(args)...);
        asm("trap;");
    }
};

/** @brief Print error message and terminate CUDA host-device function.
 *  @details Example:
 *  @code {.cu}
 *  __host__ __device__ void errorred_function {
 *      HostDevError("Error.\n");
 *  }
 *  @endcode
 *  @tparam exception Name of the exception class (like ``std::runtime_error``, ``std::invalid_argument``, etc).
 *  @param fmt Formatted string (same syntax as ``std::printf``).
 *  @param args Other arguments.
 */
template <class Exception>
struct HostDevError {
    /** @brief Constructor from formatted string.*/
    template <typename... Args>
    __cuhostdev__ HostDevError(FmtString fmt, Args &&... args) {
    #if defined(__CUDA_ARCH__)
        DeviceError(std::forward<FmtString>(fmt), std::forward<Args>(args)...);
    #else
        Fatal<Exception>(std::forward<FmtString>(fmt), std::forward<Args>(args)...);
    #endif  // __CUDA_ARCH__
    }
};

#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_LOGGER_HPP_
