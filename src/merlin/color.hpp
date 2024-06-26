// Copyright 2024 quocdang1998
#ifndef MERLIN_COLOR_HPP_
#define MERLIN_COLOR_HPP_

#include <array>  // std::array

#include "merlin/config.hpp"   // __cudevice__
#include "merlin/exports.hpp"  // MERLINENV_EXPORTS

// Terminal color
// --------------

#if !defined(__CUDA_ARCH__)
    #define __MERLIN_DEFINE_COLOR(variable_name, string_color)                                                         \
        inline constexpr const char variable_name[] = string_color
#else
    #define __MERLIN_DEFINE_COLOR(variable_name, string_color)                                                         \
        __cudevice__ constexpr const char variable_name[] = string_color
#endif

namespace merlin {

/** @brief Check if ``stdout`` is redirected into a file.*/
MERLINENV_EXPORTS extern const bool cout_terminal;

/** @brief Print out color if cout is not redirected.*/
inline constexpr const char * color_out(const char * color) { return (cout_terminal) ? color : ""; }

/** @brief Check if ``stderr`` is redirected into a file.*/
MERLINENV_EXPORTS extern const bool cerr_terminal;

/** @brief Print out color if cerr is not redirected.*/
inline constexpr const char * color_err(const char * color) { return (cerr_terminal) ? color : ""; }

/** @brief Pointer to device memory indicating if ``stdout`` is redirected into a file.*/
inline constexpr bool cuprintf_terminal = true;

/** @brief Print out color on CUDA.*/
inline constexpr const char * color_cuda(const char * color) { return (cuprintf_terminal) ? color : ""; }

namespace color {

/** @brief Switch terminal color back to normal.*/
__MERLIN_DEFINE_COLOR(normal, "\033[0m");

/** @brief Switch terminal color to bold red.*/
__MERLIN_DEFINE_COLOR(bold_red, "\033[1;31m");

/** @brief Switch terminal color to bold green.*/
__MERLIN_DEFINE_COLOR(bold_green, "\033[1;32m");

/** @brief Switch terminal color to bold yellow.*/
__MERLIN_DEFINE_COLOR(bold_yellow, "\033[1;33m");

/** @brief Switch terminal color to bold blue.*/
__MERLIN_DEFINE_COLOR(bold_blue, "\033[1;34m");

/** @brief Switch terminal color to bold magenta.*/
__MERLIN_DEFINE_COLOR(bold_magenta, "\033[1;35m");

/** @brief Switch terminal color to bold cyan.*/
__MERLIN_DEFINE_COLOR(bold_cyan, "\033[1;36m");

/** @brief Switch terminal color to bold gray.*/
__MERLIN_DEFINE_COLOR(bold_gray, "\033[1;37m");

}  // namespace color

}  // namespace merlin

#endif  // MERLIN_COLOR_HPP_
