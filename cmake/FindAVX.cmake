include(CheckCXXSourceRuns)

# source code for AVX
set(AVX_CODE
    "
#include <immintrin.h>

int main(void) {
    __m256d a = _mm256_set1_pd(0.0), b = _mm256_set1_pd(1.0), c = _mm256_set1_pd(2.0);
    c = _mm256_fmadd_pd(a, b, c);
}
")

# macro defined for checking AVX
macro(check_avx FLAG)
    set(CMAKE_REQUIRED_QUIET 1)
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS ${FLAG})
    check_cxx_source_runs("${AVX_CODE}" CXX_HAS_AVX)
    if(CXX_HAS_AVX)
        message(STATUS "Current CPU supports ${FLAG}")
    else()
        message(STATUS "Current CPU do not support ${FLAG}")
    endif()
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endmacro()

# check for AVX and save the compile option to AVX_COMPILE_OPTION
set(AVX_COMPILE_OPTION)
if(UNIX)
    check_avx("-mfma -mavx")
    if(CXX_HAS_AVX)
        set(AVX_COMPILE_OPTION -mfma -mavx)
    endif(CXX_HAS_AVX)
elseif(MSVC)
    check_avx("/arch:AVX2")
    if(CXX_HAS_AVX)
        set(AVX_COMPILE_OPTION /arch:AVX2)
    endif(CXX_HAS_AVX)
endif()
