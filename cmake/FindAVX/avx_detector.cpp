// Copyright 2024 quocdang1998
#include <iostream>  // std::cout
#include <map>       // std::map
#include <string>    // std::string

#ifdef _MSC_VER
    #include <intrin.h>  // For __cpuid on MSVC
#else
    #include <cpuid.h>  // For __get_cpuid on GCC/Clang
#endif

// Vectorization options
static std::map<unsigned int, std::string> vector_options {
    {1, ""},
#if defined(_MSC_VER)
    {2, "/arch:SSE2"},
    {4, "/arch:AVX"},
    {5, "/arch:AVX2"},
    {8, "/arch:AVX512"}
#elif defined(__GNUC__) || defined(__clang__)
    {2, "-msse2"},
    {4, "-mavx"},
    {5, "-mavx2"},
    {8, "-mavx512f"}
#endif
};

// Fused add-multiplication option
static std::map<unsigned int, std::string> fma_options {
    {0, ""},
#if defined(_MSC_VER)
    {1, ""}
#elif defined(__GNUC__) || defined(__clang__)
    {1, "-mfma"}
#endif
};

// Helper function to execute the CPUID instruction
void cpuid(int info[4], int function_id) {
#if defined(_MSC_VER)
    __cpuid(info, function_id);
#elif defined(__GNUC__) || defined(__clang__)
    __asm__ __volatile__(
        "cpuid"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(function_id), "c"(0)
    );
#endif
}

// Check if CPU supports FMA
bool supports_fma() {
    int info[4];
    cpuid(info, 1);
    return (info[2] & (1 << 12)) != 0; // FMA support is bit 12 of ECX for function_id 1
}

// Check if CPU supports SSE2
bool supports_sse2() {
    int info[4];
    cpuid(info, 1);
    return (info[3] & (1 << 26)) != 0; // SSE2 is bit 26 of EDX for function_id 1
}

// Check if CPU supports AVX
bool supports_avx() {
    int info[4];
    cpuid(info, 1);
    bool os_avx_support = (info[2] & (1 << 27)) != 0; // Check OS support (bit 27 of ECX)
    bool avx_support = (info[2] & (1 << 28)) != 0;    // AVX support (bit 28 of ECX)
    return os_avx_support && avx_support;
}

// Check if CPU supports AVX2
bool supports_avx2() {
    int info[4];
    cpuid(info, 0);
    if (info[0] >= 7) { // Check if function_id 7 is supported
        cpuid(info, 7);
        return (info[1] & (1 << 5)) != 0;  // AVX2 support (bit 5 of EBX)
    }
    return false;
}

// Check if CPU supports AVX-512
bool supports_avx512() {
    int info[4];
    cpuid(info, 0);
    if (info[0] >= 7) { // Check if function_id 7 is supported
        cpuid(info, 7);
        return (info[1] & (1 << 16)) != 0; // AVX-512F is bit 16 of EBX for function_id 7
    }
    return false;
}

int main() {
    // get vector size
    unsigned int vector_size = 1;
    if (supports_sse2()) {
        vector_size = 2;
    }
    if (supports_avx()) {
        vector_size = 4;
        if (supports_avx2()) {
            vector_size = 5;
        }
    }
    if (supports_avx512()) {
        vector_size = 8;
    }
    // get fused add-multiplication
    unsigned int fma_supports = 0;
    if (supports_fma()) {
        fma_supports = 1;
    }
    std::cout << (vector_options[vector_size] + " " + fma_options[fma_supports]);
    return 0;
}
