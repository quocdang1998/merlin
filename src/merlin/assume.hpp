// Copyright 2024 quocdang1998
#ifndef MERLIN_ASSUME_HPP_
#define MERLIN_ASSUME_HPP_

/*#if defined(__CUDA_ARCH__)
    #define assume(cond) __builtin_assume(cond)
#elif defined(_MSC_VER)
    #define assume(cond) __assume(cond)
#elif defined(__clang__)
    #define assume(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
    #define assume(cond)                                                                                               \
        do {                                                                                                           \
            if (!(cond)) __builtin_unreachable();                                                                      \
        } while (0)
#else
    #define assume(cond) ((void) 0)
#endif*/

namespace merlin {

/** @brief Compiler optimization interface for both GCC, MSVC, Clang and CUDA.*/
inline constexpr void assume(bool cond) {
#if defined(__CUDA_ARCH__)
    ::__builtin_assume(cond);
#elif defined(_MSC_VER)
    ::__assume(cond);
#elif defined(__clang__)
    ::__builtin_assume(cond);
#elif defined(__GNUC__)
    do {
        if (!(cond)) {
            ::__builtin_unreachable();
        }
    } while (0);
#endif
}

}  // namespace merlin

#endif  // MERLIN_ASSUME_HPP_
