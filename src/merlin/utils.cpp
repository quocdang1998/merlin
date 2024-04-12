// Copyright 2022 quocdang1998
#include "merlin/utils.hpp"

#include <algorithm>  // std::swap
#include <ctime>      // std::localtime, std::time, std::time_t, std::tm
#include <numeric>    // std::iota
#include <random>     // std::uniform_int_distribution
#include <sstream>    // std::ostringstream

#include "merlin/env.hpp"       // merlin::Environment
#include "merlin/logger.hpp"    // CASSERT
#include "merlin/platform.hpp"  // __MERLIN_WINDOWS__, __MERLIN_LINUX__

#if defined(__MERLIN_WINDOWS__)
    #include <windows.h>  // ::GetCurrentProcessId
#elif defined(__MERLIN_LINUX__)
    #include <unistd.h>  // ::getpid
#endif

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------------------------------------------------

// Get process ID in form of a string
std::string get_current_process_id(void) {
    std::ostringstream output;
#if defined(__MERLIN_WINDOWS__)
    output << ::GetCurrentProcessId();
#elif defined(__MERLIN_LINUX__)
    output << ::getpid();
#endif
    return output.str();
}

// Get date and time in form of a string
std::string get_time(void) {
    std::ostringstream output;
    std::time_t current_carlendar_time = std::time(nullptr);
    std::tm * now = std::localtime(&current_carlendar_time);
    output << (now->tm_year + 1900) << '-' << (now->tm_mon + 1) << '-' << now->tm_mday << '_' << now->tm_hour << ':'
           << now->tm_min << ':' << now->tm_sec;
    return output.str();
}

// ---------------------------------------------------------------------------------------------------------------------
// Random Subset
// ---------------------------------------------------------------------------------------------------------------------

// Get a random subset of index in a range
UIntVec get_random_subset(std::uint64_t num_points, std::uint64_t i_max, std::uint64_t i_min) noexcept {
    // check num_points
    if (num_points > i_max - i_min) {
        FAILURE(std::invalid_argument, "Number of points exceeding range.\n");
    }
    // calculate range index
    std::uniform_int_distribution<std::uint64_t> distribution;
    using param_t = std::uniform_int_distribution<std::uint64_t>::param_type;
    UIntVec result(num_points);
    std::iota(result.begin(), result.end(), i_min);
    std::int64_t largest_val = i_min + num_points;
    // swap with number out of range
    for (std::int64_t i = i_max - 1; i >= largest_val; i--) {
        std::uint64_t destination_idx = distribution(Environment::random_generator, param_t(i_min, i));
        if (destination_idx < largest_val) {
            result[destination_idx - i_min] = i;
        }
    }
    // swap with number in range
    for (std::int64_t i = largest_val - 1; i >= static_cast<std::int64_t>(i_min); i--) {
        std::uint64_t destination_idx = distribution(Environment::random_generator, param_t(i_min, i));
        std::swap(result[destination_idx - i_min], result[i - i_min]);
    }
    return result;
}

}  // namespace merlin
