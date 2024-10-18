// Copyright 2024 quocdang1998
#include "merlin/linalg/level1.hpp"

#include "merlin/assume.hpp"     // merlin::assume
#include "merlin/simd/simd.hpp"  // merlin:simd::Simd, merlin::simd::size, merlin::simd::alignment

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Vector-scalar arithmetic operations
// ---------------------------------------------------------------------------------------------------------------------

// Add a scalar to a vector and store the result
void linalg::add_vectors(double a, const double * b, double * c, std::uint64_t nsimd, std::uint64_t offset) noexcept {
    // hint for compiler optimizations
    assume(offset < simd::size);
    assume(reinterpret_cast<std::uintptr_t>(b) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(c) % simd::alignment == 0);
    // calculate offset
    if (offset) {
        for (std::uint64_t i = offset; i < simd::size; i++) {
            c[i] = a + b[i];
        }
        b += simd::size;
        c += simd::size;
    }
    // calculate by packs
    simd::Simd scalar(a);
    simd::Simd vector;
    for (std::uint64_t i = 0; i < nsimd; i++) {
        vector.load(b);
        vector += scalar;
        vector.store(c);
        b += simd::size;
        c += simd::size;
    }
}

// Subtract a vector from a scalar and store the result
void linalg::subtract_vectors(double a, const double * b, double * c, std::uint64_t nsimd,
                              std::uint64_t offset) noexcept {
    // hint for compiler optimizations
    assume(offset < simd::size);
    assume(reinterpret_cast<std::uintptr_t>(b) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(c) % simd::alignment == 0);
    // calculate offset
    if (offset) {
        for (std::uint64_t i = offset; i < simd::size; i++) {
            c[i] = a - b[i];
        }
        b += simd::size;
        c += simd::size;
    }
    // calculate by packs
    simd::Simd scalar(a);
    simd::Simd vector;
    for (std::uint64_t i = 0; i < nsimd; i++) {
        vector.load(b);
        vector = scalar - vector;
        vector.store(c);
        b += simd::size;
        c += simd::size;
    }
}

// Subtract a scalar from a vector and store the result
void linalg::subtract_vectors(const double * a, double b, double * c, std::uint64_t nsimd,
                                     std::uint64_t offset) noexcept {
    // hint for compiler optimizations
    assume(offset < simd::size);
    assume(reinterpret_cast<std::uintptr_t>(a) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(c) % simd::alignment == 0);
    // calculate offset
    if (offset) {
        for (std::uint64_t i = offset; i < simd::size; i++) {
            c[i] = a[i] - b;
        }
        a += simd::size;
        c += simd::size;
    }
    // calculate by packs
    simd::Simd scalar(b);
    simd::Simd vector;
    for (std::uint64_t i = 0; i < nsimd; i++) {
        vector.load(a);
        vector -= scalar;
        vector.store(c);
        a += simd::size;
        c += simd::size;
    }
}

// Multiply a scalar by a vector and store the result
void linalg::multiply_vectors(double a, const double * b, double * c, std::uint64_t nsimd,
                              std::uint64_t offset) noexcept {
    // hint for compiler optimizations
    assume(offset < simd::size);
    assume(reinterpret_cast<std::uintptr_t>(b) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(c) % simd::alignment == 0);
    // calculate offset
    if (offset) {
        for (std::uint64_t i = offset; i < simd::size; i++) {
            c[i] = a * b[i];
        }
        b += simd::size;
        c += simd::size;
    }
    // calculate by packs
    simd::Simd scalar(a);
    simd::Simd vector;
    for (std::uint64_t i = 0; i < nsimd; i++) {
        vector.load(b);
        vector *= scalar;
        vector.store(c);
        b += simd::size;
        c += simd::size;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Element wise operations
// ---------------------------------------------------------------------------------------------------------------------

// Add two vectors and store the result
void linalg::add_vectors(const double * a, const double * b, double * c, std::uint64_t npacks,
                         std::uint64_t offset) noexcept {
    // hint for compiler optimizations
    assume(offset < simd::size);
    assume(reinterpret_cast<std::uintptr_t>(a) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(b) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(c) % simd::alignment == 0);
    // calculate offset
    if (offset) {
        for (std::uint64_t i = offset; i < simd::size; i++) {
            c[i] = a[i] + b[i];
        }
        a += simd::size;
        b += simd::size;
        c += simd::size;
    }
    // calculate by packs
    simd::Simd vector_sum;
    simd::Simd vector_adden;
    for (std::uint64_t i = 0; i < npacks; i++) {
        vector_sum.load(a);
        vector_adden.load(b);
        vector_sum += vector_adden;
        vector_sum.store(c);
        a += simd::size;
        b += simd::size;
        c += simd::size;
    }
}

// Subtract two vectors and store the result
void subtract_vectors(const double * a, const double * b, double * c, std::uint64_t npacks,
                      std::uint64_t offset) noexcept {
    // hint for compiler optimizations
    assume(offset < simd::size);
    assume(reinterpret_cast<std::uintptr_t>(a) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(b) % simd::alignment == 0);
    assume(reinterpret_cast<std::uintptr_t>(c) % simd::alignment == 0);
    // calculate offset
    if (offset) {
        for (std::uint64_t i = offset; i < simd::size; i++) {
            c[i] = a[i] - b[i];
        }
        a += simd::size;
        b += simd::size;
        c += simd::size;
    }
    // calculate by packs
    simd::Simd vector_diff;
    simd::Simd vector_subtrahend;
    for (std::uint64_t i = 0; i < npacks; i++) {
        vector_diff.load(a);
        vector_subtrahend.load(b);
        vector_diff -= vector_subtrahend;
        vector_diff.store(c);
        a += simd::size;
        b += simd::size;
        c += simd::size;
    }
}

}  // namespace merlin
