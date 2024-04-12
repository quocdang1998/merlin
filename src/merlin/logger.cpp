// Copyright quocdang1998
#include "merlin/logger.hpp"

#include <algorithm>  // std::copy

#if defined(__MERLIN_WINDOWS__)
    #include <windows.h>  // ::FormatMessageA, ::GetCurrentProcess
    #ifdef __MERLIN_DEBUG__
        #include <cstdint>    // std::uint64_t
        #include <cstdlib>    // std::calloc
        #include <cstring>    // std::strlen
        #include <dbghelp.h>  // ::CaptureStackBackTrace, ::SymInitialize, ::SymFromAddr, ::SYMBOL_INFO
    #endif                    // __MERLIN_DEBUG__
#elif defined(__MERLIN_LINUX__)
    #include <errno.h>   // errno
    #include <string.h>  // ::strerror
    #ifdef __MERLIN_DEBUG__
        #include <cxxabi.h>    // ::abi::__cxa_demangle
        #include <dlfcn.h>     // ::Dl_info, ::dladdr
        #include <execinfo.h>  // ::backtrace
    #endif                     // __MERLIN_DEBUG__
#endif

// Max depth of the stacktrace
#define STACK_TRACE_BUFFER_SIZE 128

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Get error message and print stacktrace
// ---------------------------------------------------------------------------------------------------------------------

// Pointer to frame
typedef void * native_frame_ptr_t;

#if defined(__MERLIN_WINDOWS__)

// Get error from Windows API
std::string throw_windows_last_error(unsigned long int last_error) {
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
    #ifdef __MERLIN_DEBUG__
    // get current process
    native_frame_ptr_t process = ::GetCurrentProcess();
    ::SymInitialize(process, nullptr, true);
    // capture address of the functions in the stacktrace
    native_frame_ptr_t buffer[STACK_TRACE_BUFFER_SIZE];
    unsigned int frames = ::CaptureStackBackTrace(skip, STACK_TRACE_BUFFER_SIZE, buffer, nullptr);
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
    #endif  // __MERLIN_DEBUG__
}

#elif defined(__MERLIN_LINUX__)

// Get error from Linux
std::string throw_linux_last_error(void) {
    if (errno != 0) {
        char * buffer = ::strerror(errno);
        return std::string(buffer);
    } else {
        return std::string();
    }
}

// Print stacktrace
void print_stacktrace(int skip) {
    #ifdef __MERLIN_DEBUG__
    // get number of frame in the stack
    native_frame_ptr_t buffer[STACK_TRACE_BUFFER_SIZE];
    int frames_count = ::backtrace(const_cast<void **>(buffer), STACK_TRACE_BUFFER_SIZE);
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
    #endif  // __MERLIN_DEBUG__
}

#endif

}  // namespace merlin
