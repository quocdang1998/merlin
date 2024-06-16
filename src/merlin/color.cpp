// Copyright 2024 quocdang1998
#include "merlin/color.hpp"

#include "merlin/platform.hpp"  // __MERLIN_LINUX__, __MERLIN_WINDOWS__

#if defined(__MERLIN_WINDOWS__)
    #include <io.h>     // ::_isatty
    #include <stdio.h>  // ::_fileno, stdout, stderr
#elif defined(__MERLIN_LINUX__)
    #include <cstdio>    // stdout, stderr
    #include <unistd.h>  // ::fileno, ::isatty
#endif

namespace merlin {

#if defined(__MERLIN_WINDOWS__)
const bool cout_terminal = ::_isatty(::_fileno(stdout));
const bool cerr_terminal = ::_isatty(::_fileno(stderr));
#elif defined(__MERLIN_LINUX__)
const bool cout_terminal = ::isatty(::fileno(stdout));
const bool cerr_terminal = ::isatty(::fileno(stderr));
#endif

}  // namespace merlin
