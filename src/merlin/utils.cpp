// Copyright 2022 quocdang1998
#include "merlin/utils.hpp"

#include <algorithm>  // std::shuffle
#include <ctime>      // std::localtime, std::time, std::time_t, std::tm
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
intvec get_random_subset(std::uint64_t num_points, std::uint64_t i_max, std::uint64_t i_min) noexcept {
    // check num_points
    CASSERT(num_points > i_max - i_min, FAILURE, std::invalid_argument, "Number of points exceeding range.\n");
    // calculate range index
    intvec range_index(i_max - i_min);
    for (std::uint64_t i = 0; i < range_index.size(); i++) {
        range_index[i] = i_min + i;
    }
    std::shuffle(range_index.begin(), range_index.end(), Environment::random_generator);
    // return a few first index
    return intvec(range_index.data(), num_points);
}

}  // namespace merlin
