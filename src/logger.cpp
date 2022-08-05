// Copyright 2022 quocdang1998
#include "merlin/logger.hpp"

#include <cstdarg>
#include <stdexcept>

void logger_(LogMode mode, const std::string & func_name, const char * fmt, ...) {
    // treat formatted string
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
        case LogMode::Failure:
            std::fprintf(stderr, "\033[1;31m[FAILURE]\033[0m [%s] %s\n", func_name.c_str(), buffer);
            throw std::runtime_error(const_cast<char *>(buffer));
            break;
    }
}
