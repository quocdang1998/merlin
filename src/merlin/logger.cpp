// Copyright quocdang1998
#include "merlin/logger.hpp"

#include <algorithm>  // std::copy

#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__

#if defined(__MERLIN_WINDOWS__)
    #include <windows.h>  // ::FormatMessageA, ::GetCurrentProcess
    #ifndef NDEBUG
        #include <cstdint>    // std::uint64_t
        #include <cstdlib>    // std::calloc
        #include <cstring>    // std::strlen
        #include <dbghelp.h>  // ::CaptureStackBackTrace, ::SymInitialize, ::SymFromAddr, ::SYMBOL_INFO
    #endif
#elif defined(__MERLIN_LINUX__)
    #include <errno.h>   // errno
    #include <string.h>  // ::strerror
    #ifndef NDEBUG
        #include <cxxabi.h>    // ::abi::__cxa_demangle
        #include <dlfcn.h>     // ::Dl_info, ::dladdr
        #include <execinfo.h>  // ::backtrace
    #endif
#endif

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Stack Tracing
// ---------------------------------------------------------------------------------------------------------------------

// Pointer to frame
using native_frame_ptr_t = void *;

// Max depth of the stacktrace
inline constexpr std::uint64_t stacktrace_buffer = 128;

#if defined(__MERLIN_WINDOWS__)

// Get error from Windows API
std::string throw_sys_last_error(unsigned long int last_error) {
    if (last_error != 0) {
        char * buffer = nullptr;
        const unsigned long int format = FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                         FORMAT_MESSAGE_IGNORE_INSERTS;
        unsigned long int size = ::FormatMessageA(format, nullptr, last_error,
                                                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                                  reinterpret_cast<char *>(&buffer), 0, nullptr);
        return std::string(buffer, size);
    } else {
        return std::string();
    }
}

// Print stacktrace
void print_stacktrace(int skip) {
    #ifndef NDEBUG
    // get current process
    native_frame_ptr_t process = ::GetCurrentProcess();
    ::SymInitialize(process, nullptr, true);
    // capture address of the functions in the stacktrace
    native_frame_ptr_t buffer[stacktrace_buffer];
    unsigned int frames = ::CaptureStackBackTrace(skip, stacktrace_buffer, buffer, nullptr);
    ::SYMBOL_INFO * symbol = static_cast<::SYMBOL_INFO *>(std::calloc(sizeof(::SYMBOL_INFO) + 256 * sizeof(char), 1));
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(::SYMBOL_INFO);
    // print symbol name
    std::fprintf(stderr, "Stack backtrace:\n");
    for (unsigned int i = 0; i < frames; i++) {
        ::SymFromAddr(process, reinterpret_cast<std::uint64_t>(buffer[i]), 0, symbol);
        if (std::strlen(symbol->Name)) {
            std::fprintf(stderr, "    %s\n", symbol->Name);
        } else {
            std::fprintf(stderr, "    %p\n", buffer[i]);
        }
    }
    std::free(symbol);
    #endif
}

#elif defined(__MERLIN_LINUX__)

// Get error from Linux
std::string throw_sys_last_error(unsigned long int last_error) {
    if (errno != 0) {
        char * buffer = ::strerror(errno);
        return std::string(buffer);
    } else {
        return std::string();
    }
}

// Print stacktrace
void print_stacktrace(int skip) {
    #ifndef NDEBUG
    // get number of frame in the stack
    native_frame_ptr_t buffer[stacktrace_buffer];
    int frames_count = ::backtrace(const_cast<void **>(buffer), stacktrace_buffer);
    std::copy(buffer + skip, buffer + frames_count, buffer);
    frames_count -= skip;
    if (frames_count && buffer[frames_count - 1] == nullptr) {
        --frames_count;
    }
    // get name (demangled) for each frame
    std::fprintf(stderr, "Stack backtrace:\n");
    for (int i = 0; i < frames_count; i++) {
        ::Dl_info dli;
        bool dl_ok = !!::dladdr(const_cast<void *>(buffer[i]), &dli);
        // if the symbolic name can be retrieved, print mangled name, else print function address
        if (dl_ok && dli.dli_sname) {
            int status = 0;
            std::size_t size = 0;
            char * demangled_name = ::abi::__cxa_demangle(dli.dli_sname, NULL, &size, &status);
            // if demangle success
            if (demangled_name != nullptr) {
                std::fprintf(stderr, "    %s\n", demangled_name);
                free(demangled_name);
            } else {
                std::fprintf(stderr, "    %s\n", dli.dli_sname);
            }
        } else {
            std::fprintf(stderr, "    %p\n", buffer[i]);
        }
    }
    #endif
}

#endif

}  // namespace merlin
