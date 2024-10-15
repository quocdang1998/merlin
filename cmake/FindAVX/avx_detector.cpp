#include <immintrin.h>
#include <iostream>

#ifdef _MSC_VER
    #include <intrin.h>  // For __cpuid on MSVC
#else
    #include <cpuid.h>  // For __get_cpuid on GCC/Clang
#endif

// Check if CPU supports AVX
bool cpu_supports_avx() {
    // initialize registers
    unsigned int eax, ebx, ecx, edx;
    // retrieve the ECX register
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    ecx = cpu_info[2];
#else
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
#endif
    // check if AVX is supported usign the bit 28-th of ECX (CPUID.01H:ECX.AVX[bit 28])
    bool os_uses_xsave_xrstor = (ecx & (1 << 27)) != 0;
    bool avx_supported = (ecx & (1 << 28)) != 0;

    // check if XGETBV instruction supports AVX by checking XCR0 register
    if (os_uses_xsave_xrstor && avx_supported) {
        unsigned long long xcr_feature_mask = _xgetbv(0);
        return (xcr_feature_mask & 0x6) == 0x6;
    }
    return false;
}

// Check if CPU supports AVX2
bool cpu_supports_avx2() {
    // initialize registers
    unsigned int eax, ebx, ecx, edx;
    // retrieve the EBX register
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    ebx = cpu_info[1];
#else
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);
#endif
    // check AVX2 support using the bit 5-th of EBX (CPUID.07H:EBX.AVX2[bit 5])
    return (ebx & (1 << 5)) != 0;
}

// Check if CPU supports AVX-512
bool cpu_supports_avx512() {
    // initialize registers
    unsigned int eax, ebx, ecx, edx;
    // retrieve the EBX register
#ifdef _MSC_VER
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    ebx = cpu_info[1];
#else
    __get_cpuid(7, &eax, &ebx, &ecx, &edx);
#endif
    // check AVX-512 support using the bit 16-th of EBX (CPUID.07H:EBX.AVX512F[bit 16])
    return (ebx & (1 << 16)) != 0;
}

int main() {
    unsigned int vector_size = 1;
    if (cpu_supports_avx()) {
        vector_size = 2;
    }
    if (cpu_supports_avx2()) {
        vector_size = 4;
    }
    if (cpu_supports_avx512()) {
        vector_size = 8;
    }
    std::cout << vector_size;
    return 0;
}
