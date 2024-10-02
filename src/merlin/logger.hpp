// Copyright 2022 quocdang1998
#ifndef MERLIN_LOGGER_HPP_
#define MERLIN_LOGGER_HPP_

#include <cinttypes>        // PRIdLEAST32
#include <cstdio>           // std::fprintf, std::snprintf
#include <filesystem>       // std::filesystem::filesystem_error
#include <iostream>         // std::cout, std::clog
#include <source_location>  // std::source_location
#include <stdexcept>        // std::runtime_error
#include <syncstream>       // std::osyncstream
#include <utility>          // std::forward
#include <format>

#include "merlin/color.hpp"     // __MERLIN_COLOR
#include "merlin/config.hpp"    // merlin::printf_buffer, __cudevice__, __cuhostdev__
#include "merlin/exports.hpp"   // MERLINENV_EXPORTS
#include "merlin/platform.hpp"  // __MERLIN_LINUX__

namespace merlin {

// Stack tracing
// -------------

/** @brief Print the stacktrace at the crash moment.*/
MERLINENV_EXPORTS void print_stacktrace(int skip = 1);

// Get error message from Windows and Unix
// ---------------------------------------

/** @brief Get error message from Windows or Linux.*/
MERLINENV_EXPORTS std::string throw_sys_last_error(unsigned long int last_error = static_cast<unsigned long int>(-1));

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
 *  Message("A message with a value %d.\n", -2);
 *  // use case 2: multiple output
 *  Message m("A printf ");
 *  m << "that takes " << "extra arguments. ";
 *  m << "And this message will not be mixed up even in multi-threading.";
 *  @endcode
 */
struct Message {
  public:
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    Message(FmtString fmt = "", Args &&... args) {
        char format_buffer[printf_buffer] = "%smessage%s [%s] in %s l.%" PRIdLEAST32 " : ";
        cat_str(format_buffer + len(format_buffer), fmt.str);
        char buffer[printf_buffer];
        std::snprintf(buffer, printf_buffer, format_buffer, color_out(color::bold_blue), color_out(color::normal),
                      fmt.loc.function_name(), get_fname(fmt.loc.file_name()), fmt.loc.line(),
                      std::forward<Args>(args)...);
        this->stream << buffer;
    }
    /** @brief Stream operator.*/
    template <typename T>
    Message & operator<<(T && obj) {
        this->stream << std::forward<T>(obj);
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
 *  Warning("A warning with a value %f.\n", 0.5);
 *  // use case 2: multiple output
 *  Warning w("A warning ");
 *  w << "that takes " << "extra arguments. ";
 *  w << "And this warning will not be mixed up even in multi-threading.";
 *  @endcode
 */
struct Warning {
  public:
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    Warning(FmtString fmt = "", Args &&... args) {
        char format_buffer[printf_buffer] = "%swarning%s [%s] in %s l.%" PRIdLEAST32 " : ";
        cat_str(format_buffer + len(format_buffer), fmt.str);
        char buffer[printf_buffer];
        std::snprintf(buffer, printf_buffer, format_buffer, color_err(color::bold_yellow), color_err(color::normal),
                      fmt.loc.function_name(), get_fname(fmt.loc.file_name()), fmt.loc.line(),
                      std::forward<Args>(args)...);
        this->stream << buffer;
    }
    /** @brief Stream operator.*/
    template <typename T>
    Warning & operator<<(T && obj) {
        this->stream << std::forward<T>(obj);
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
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     */
    DebugLog(FmtString fmt, Args &&... args) {
#ifndef NDEBUG
        char buffer[printf_buffer] = "%sdebug%s [%s] in %s l.%" PRIdLEAST32 " : ";
        cat_str(buffer + len(buffer), fmt.str);
        std::fprintf(stderr, buffer, color_err(color::bold_green), color_err(color::normal), fmt.loc.function_name(),
                     get_fname(fmt.loc.file_name()), fmt.loc.line(), std::forward<Args>(args)...);
        print_stacktrace(2);
#endif
    }
};

/** @brief Print error message and throw an instance of type exception.
 *  @details Example:
 *  @code {.cpp}
 *  Fatal<std::runtime_error>("A runtime error message.\n");
 *  @endcode
 *  @note The use of this class is recommended over C++ exception throwing, because some compiler/debugger may be unable
 *  to display automatically the error message.
 *  @tparam Exception Name of the exception class (like ``std::runtime_error``, ``std::invalid_argument``, etc).
 */
template <class Exception>
struct Fatal {
  public:
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    Fatal(FmtString fmt, Args &&... args) {
        // create exception message
        char exception_buffer[printf_buffer];
#if defined(__MERLIN_LINUX__)
        _Pragma("GCC diagnostic ignored \"-Wformat-security\"");
#endif  // __MERLIN_LINUX__
        std::snprintf(exception_buffer, printf_buffer, fmt.str, std::forward<Args>(args)...);
#if defined(__MERLIN_LINUX__)
        _Pragma("GCC diagnostic pop");
#endif  // __MERLIN_LINUX__
        // concatenate exception message to source location info
        std::fprintf(stderr, "%sfatal%s [%s] in %s l.%" PRIdLEAST32 " : %s", color_err(color::bold_red),
                     color_err(color::normal), fmt.loc.function_name(), get_fname(fmt.loc.file_name()), fmt.loc.line(),
                     exception_buffer);
#ifndef NDEBUG
        print_stacktrace(2);
#endif
        if constexpr (std::is_same<Exception, std::filesystem::filesystem_error>::value) {
            throw std::filesystem::filesystem_error(exception_buffer, std::error_code(1, std::iostream_category()));
        } else {
            throw Exception(exception_buffer);
        }
    }
};

// Log CudaOut and CudaDeviceError for GPU
// ---------------------------------------

#if defined(__NVCC__)

/** @brief Print message to the standard output (for usage inside a GPU function).
 *  @details Example:
 *  @code {.cu}
 *  __global__ void dummy_gpu_function {
 *      CudaOut("Hello World from thread %d block %d.\n", threadIdx.x, blockIdx.x);
 *  }
 *  @endcode
 *  @note This class is only available on device code.
 */
struct CudaOut {
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    __cudevice__ CudaOut(FmtString fmt, Args &&... args) {
        char buffer[printf_buffer] = "%scudaout%s [%s] in %s l.%" PRIdLEAST32 " : ";
        cat_str(buffer + len(buffer), fmt.str);
        std::printf(buffer, color_cuda(color::bold_cyan), color_cuda(color::normal), fmt.loc.function_name(),
                    get_fname(fmt.loc.file_name()), fmt.loc.line(), std::forward<Args>(args)...);
    }
};

/** @brief Print error message (for usage inside a GPU function) and terminate GPU kernel.
 *  @details Example:
 *  @code {.cu}
 *  __global__ void error_gpu_function {
 *      DeviceError("Error from thread %d block %d.\n", threadIdx.x, blockIdx.x);
 *  }
 *  @endcode
 *  @note This class is only available on device code.
 */
struct DeviceError {
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    __cudevice__ DeviceError(FmtString fmt, Args &&... args) {
        char buffer[printf_buffer] = "cudaerror [%s] in %s l.%" PRIdLEAST32 " : ";
        cat_str(buffer + len(buffer), fmt.str);
        std::printf(buffer, fmt.loc.function_name(), get_fname(fmt.loc.file_name()), fmt.loc.line(),
                    std::forward<Args>(args)...);
        asm("trap;");
    }
};

#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_LOGGER_HPP_
