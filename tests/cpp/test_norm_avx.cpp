#include "merlin/linalg/dot.hpp"

#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

void generate(std::vector<double> & v, unsigned long seed) {
    std::mt19937 gen{seed};
    std::uniform_real_distribution<> dist{-1.0, 1.0};
    for (int i = 0; i < v.size(); i++) {
        v[i] = dist(gen);
    }
}

int main(void) {
    using namespace merlin;

    // initialize
    std::vector<double> a(25962), b(25962);
    generate(a, 1);
    generate(b, 2);
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    double result1, result2;

    // calculation time using regular O3 optimization
    start = std::chrono::high_resolution_clock::now();
    // linalg::__norm_no_avx(a.data(), a.size(), result1);
    linalg::__dot_no_avx(a.data(), b.data(), a.size(), result1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(25) << "Baseline Elapsed Time: " << std::setw(8)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              << " ns, result " << std::setprecision(18) << result1 << std::endl;

    // calculation time using AVX
    start = std::chrono::high_resolution_clock::now();
    // linalg::__norm_256_avx(a.data(), a.size(), result2);
    linalg::__dot_256_avx(a.data(), b.data(), a.size(), result2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << std::setw(25) << "AVX2-FMA Elapsed Time: " << std::setw(8)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
              << " ns, result " << std::setprecision(18) << result2 << std::endl;

    std::cout << "  Is identical: " << std::boolalpha << (result1 == result2) << std::endl;
}
