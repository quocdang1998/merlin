// Copyright 2022 quocdang1998
#ifndef MERLIN_LOGGER_HPP_
#define MERLIN_LOGGER_HPP_

#include <cstdio>
#include <string>
#include <cstdarg>
#include <stdexcept>

// define log messages for GNU g++ on Linux
#if defined(__GNUG__)
#define MESSAGE(fmt, ...) logger_(LogMode::Message, __PRETTY_FUNCTION__, fmt, ## __VA_ARGS__)
#define WARNING(fmt, ...) logger_(LogMode::Warning, __PRETTY_FUNCTION__, fmt, ##__VA_ARGS__)
#define FAILURE(fmt, ...) error_(__PRETTY_FUNCTION__, fmt, ##__VA_ARGS__)

// define log messages for MSVC cl.exe on Windows
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define MESSAGE(fmt, ...) logger_(LogMode::Message, __FUNCSIG__, fmt, ##__VA_ARGS__)
#define WARNING(fmt, ...) logger_(LogMode::Warning, __FUNCSIG__, fmt, ##__VA_ARGS__)
#define FAILURE(exception, fmt, ...) error_<exception>(__FUNCSIG__, fmt, ##__VA_ARGS__)
#endif

// Log mode
enum class LogMode {
    Message,
    Warning,
};

// print log for MESSAGE and WARNING
inline void logger_(LogMode mode, const std::string & func_name, const char * fmt, ...) {
    // save formatted string to a buffer
    char buffer[1024];
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    // format message and print
    switch (mode) {
        case LogMode::Message:
            std::printf("\033[1;34m[MESSAGE]\033[0m [%s] %s\n", func_name.c_str(), buffer);
            break;
        case LogMode::Warning:
            std::fprintf(stderr, "\033[1;33m[WARNING]\033[0m [%s] %s\n", func_name.c_str(), buffer);
            break;
    }
}

// print log for FAILURE + exception
template <class Exception = std::runtime_error>
void error_(const std::string & func_name, const char * fmt, ...) {
    // save formatted string to a buffer
    char buffer[1024];
    std::va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    // print exception message and throw corresponding exception
    std::fprintf(stderr, "\033[1;31m[FAILURE]\033[0m [%s] %s\n", func_name.c_str(), buffer);
    throw Exception(const_cast<char *>(buffer));
}

#endif  // MERLIN_LOGGER_HPP_
