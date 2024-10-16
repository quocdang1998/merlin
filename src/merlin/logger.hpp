// Copyright 2022 quocdang1998
#ifndef MERLIN_LOGGER_HPP_
#define MERLIN_LOGGER_HPP_

#include <algorithm>        // std::copy_n
#include <cinttypes>        // PRIdLEAST32
#include <cstdio>           // std::printf
#include <filesystem>       // std::filesystem::filesystem_error
#include <format>           // std::format, std::vformat
#include <iostream>         // std::cout, std::clog
#include <ostream>          // std::ostream
#include <source_location>  // std::source_location
#include <stdexcept>        // std::runtime_error
#include <syncstream>       // std::osyncstream
#include <utility>          // std::forward

#include "merlin/color.hpp"    // merlin::color, merlin::color_out, merlin::color_err
#include "merlin/config.hpp"   // __cudevice__
#include "merlin/exports.hpp"  // MERLIN_EXPORTS

namespace merlin {

// Stack tracing
// -------------

/** @brief Print the stacktrace at the crash moment.*/
MERLIN_EXPORTS void print_stacktrace(std::ostream & output, int skip = 1);

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

// Log Message, Warning, DebugLog and Fatal for CPU
// ------------------------------------------------

/** @brief Compile-time function to calculate string length.*/
constexpr std::size_t get_length(const char * str) {
    std::size_t len = 0;
    while (str[len] != '\0') {
        ++len;
    }
    return len;
}

/** @brief Compile-time function to extract the filename from a file path.*/
constexpr const char * extract_filename(const char * path) {
    const std::size_t len = get_length(path);
    const char * last_slash = path;
    for (std::size_t i = 0; i < len; ++i) {
        if (path[i] == '/' || path[i] == '\\') {
            last_slash = &path[i + 1];
        }
    }
    return last_slash;
}

/** @brief Wrapper for format string and current source location.*/
struct FmtString {
    /** @brief Constructor.*/
    template <std::size_t N>
    constexpr FmtString(const char (&data)[N],
                        const std::source_location & location = std::source_location::current()) :
    str{data}, size{N}, loc{location}, filename{extract_filename(location.file_name())} {}

    /** @brief Pointer to the associated message.*/
    const char * const str;
    /** @brief Size of the string.*/
    const std::size_t size;
    /** @brief Location of the source.*/
    const std::source_location & loc;
    /** @brief Filename.*/
    const std::string_view filename;
};

/** @brief Print message to the standard output.
 *  @details Example:
 *  @code {.cpp}
 *  // use case 1: single output
 *  Message("A message with a value {}.\n", -2);
 *  // use case 2: multiple output
 *  Message m("A message ");
 *  m << "that takes " << "extra arguments. ";
 *  m << "And this message will not be mixed up even in multi-threading.";
 *  @endcode
 */
struct Message {
  public:
    /** @brief Constructor from formatting string and variadic template arguments.
     *  @param fmt Format string (same syntax as ``std::format``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    Message(const FmtString & fmt, Args &&... args) {
#if !defined(__CUDA_ARCH__)
        this->stream << std::format(Message::prefix, color_out(color::bold_blue), color_out(color::normal),
                                    fmt.loc.function_name(), fmt.filename, fmt.loc.line());
        this->stream << std::vformat(std::string_view(fmt.str, fmt.size), std::make_format_args(args...));
#endif  // __CUDA_ARCH__
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
    /** @brief Associated asynchronous stream.*/
    std::osyncstream stream{std::cout};
    /** @brief Appended prefix.*/
    static constexpr const char prefix[] = "{}message{} [{}] in {} l.{} : ";
};

/** @brief Print warning to the standard error.
 *  @details Example:
 *  @code {.cpp}
 *  // use case 1: single output
 *  Warning("A warning with a value {}.\n", 0.5);
 *  // use case 2: multiple output
 *  Warning w("A warning ");
 *  w << "that takes " << "extra arguments. ";
 *  w << "And this warning will not be mixed up even in multi-threading.";
 *  @endcode
 */
struct Warning {
  public:
    /** @brief Constructor from formatting string and variadic template arguments.
     *  @param fmt Format string (same syntax as ``std::format``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    Warning(const FmtString & fmt, Args &&... args) {
#if !defined(__CUDA_ARCH__)
        this->stream << std::format(Warning::prefix, color_out(color::bold_yellow), color_out(color::normal),
                                    fmt.loc.function_name(), fmt.filename, fmt.loc.line());
        this->stream << std::vformat(std::string_view(fmt.str, fmt.size), std::make_format_args(args...));
#endif  // __CUDA_ARCH__
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
    /** @brief Associated asynchronous stream.*/
    std::osyncstream stream{std::clog};
    /** @brief Appended prefix.*/
    static constexpr const char prefix[] = "{}warning{} [{}] in {} l.{} : ";
};

/** @brief Log messages that is printed only in debug mode.
 *  @details Example:
 *  @code {.cpp}
 *  DebugLog("A debug log with a value {} that is only printed in debug mode.\n", 0.5);
 *  @endcode
 */
struct DebugLog {
  public:
    /** @brief Constructor from formatting string and variadic template arguments.
     *  @param fmt Format string (same syntax as ``std::format``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    DebugLog(const FmtString & fmt, Args &&... args) {
#if !defined(__CUDA_ARCH__) && !defined(NDEBUG)
        this->stream << std::format(DebugLog::prefix, color_out(color::bold_green), color_out(color::normal),
                                    fmt.loc.function_name(), fmt.filename, fmt.loc.line());
        this->stream << std::vformat(std::string_view(fmt.str, fmt.size), std::make_format_args(args...));
#endif  // !__CUDA_ARCH__ && !NDEBUG
        print_stacktrace(this->stream, 2);
    }
    /** @brief Stream operator.*/
    template <typename T>
    DebugLog & operator<<(T && obj) {
        this->stream << std::forward<T>(obj);
        return *this;
    }
    /** @brief Force print out content to the output stream.*/
    void emit(void) { this->stream.emit(); }

  private:
    /** @brief Associated asynchronous stream.*/
    std::osyncstream stream{std::clog};
    /** @brief Appended prefix.*/
    static constexpr const char prefix[] = "{}debug{} [{}] in {} l.{} : ";
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
    /** @brief Throw default constructed exception.*/
    Fatal(const std::source_location & loc = std::source_location::current()) {
#if !defined(__CUDA_ARCH__)
        this->stream << std::format(Fatal::default_prefix, color_out(color::bold_red), color_out(color::normal),
                                    loc.function_name(), extract_filename(loc.file_name()), loc.line());
#endif  // __CUDA_ARCH__
        print_stacktrace(this->stream, 2);
        this->stream.emit();
        throw Exception();
    }
    /** @brief @brief Constructor from formatting string and variadic template arguments.
     *  @param fmt Format string (same syntax as ``std::format``).
     *  @param args Arguments to be formatted.
     */
    template <typename... Args>
    Fatal(const FmtString & fmt, Args &&... args) {
        std::string error_message;
#if !defined(__CUDA_ARCH__)
        error_message = std::vformat(std::string_view(fmt.str, fmt.size), std::make_format_args(args...));
        this->stream << std::format(Fatal::prefix, color_out(color::bold_red), color_out(color::normal),
                                    fmt.loc.function_name(), fmt.filename, fmt.loc.line());
#endif  // __CUDA_ARCH__
        this->stream << error_message;
        print_stacktrace(this->stream, 2);
        this->stream.emit();
        if constexpr (std::is_same<Exception, std::filesystem::filesystem_error>::value) {
            throw std::filesystem::filesystem_error(error_message.c_str(),
                                                    std::error_code(1, std::iostream_category()));
        } else {
            throw Exception(error_message.c_str());
        }
    }

  private:
    /** @brief Associated asynchronous stream.*/
    std::osyncstream stream{std::clog};
    /** @brief Appended prefix.*/
    static constexpr const char prefix[] = "{}fatal{} [{}] in {} l.{} : ";
    /** @brief Default prefix for the case without message.*/
    static constexpr const char default_prefix[] = "{}fatal{} [{}] in {} l.{}";
};

// Log CudaOut and CudaDeviceError for GPU
// ---------------------------------------

/** @brief Joint to string literals into a compile-time array.*/
template <std::size_t N1, std::size_t N2>
constexpr auto join_str(const char (&s1)[N1], const char (&s2)[N2]) {
    std::array<char, N1 + N2 - 1> result;
    std::copy_n(s1, N1 - 1, result.data());
    std::copy_n(s2, N2, result.data() + N1 - 1);
    return result;
}

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
template <std::size_t N, typename... Args>
struct CudaOut {
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     *  @param loc SOurce location.
     */
    __cudevice__ CudaOut(const char (&fmt)[N], Args &&... args,
                         const std::source_location & loc = std::source_location::current()) {
        auto str = join_str("%scudaout%s [%s] in %s l.%" PRIdLEAST32 " : ", fmt);
        std::printf(str.data(), color::bold_cyan, color::normal, loc.function_name(), extract_filename(loc.file_name()),
                    loc.line(), std::forward<Args>(args)...);
    }
};
#if !defined(__DOXYGEN_PARSER__)
template <std::size_t N, typename... Args>
CudaOut(const char (&)[N], Args &&...) -> CudaOut<N, Args...>;
#endif  // !__DOXYGEN_PARSER__


/** @brief Print error message (for usage inside a GPU function) and terminate GPU kernel.
 *  @details Example:
 *  @code {.cu}
 *  __global__ void error_gpu_function {
 *      DeviceError("Error from thread %d block %d.\n", threadIdx.x, blockIdx.x);
 *  }
 *  @endcode
 *  @note This class is only available on device code.
 */
template <std::size_t N, typename... Args>
struct DeviceError {
    /** @brief Constructor from ``std::printf``'s syntax.
     *  @param fmt Format string (same syntax as ``std::printf``).
     *  @param args Arguments to be formatted.
     *  @param loc SOurce location.
     */
    __cudevice__ DeviceError(const char (&fmt)[N], Args &&... args,
                             const std::source_location & loc = std::source_location::current()) {
        auto str = join_str("%cudaerr%s [%s] in {%s} l.%" PRIdLEAST32 " : ", fmt);
        std::printf(str.data(), color::bold_magenta, color::normal, loc.function_name(),
                    extract_filename(loc.file_name()), loc.line(), std::forward<Args>(args)...);
        asm("trap;");
    }
};
#if !defined(__DOXYGEN_PARSER__)
template <std::size_t N, typename... Args>
DeviceError(const char (&)[N], Args &&...) -> DeviceError<N, Args...>;
#endif  // !__DOXYGEN_PARSER__

#endif  // __NVCC__

}  // namespace merlin

#endif  // MERLIN_LOGGER_HPP_
