// Copyright 2023 quocdang1998
#ifndef MERLIN_CANDY_PARTITION_HPP_
#define MERLIN_CANDY_PARTITION_HPP_

#include <cstdint>  // std::uint64_t

#include "merlin/exports.hpp"  // MERLIN_EXPORTS
#include "merlin/vector.hpp"   // merlin::Vector, merlin::intvec

namespace merlin {

namespace candy {

/** @brief Create partition index from contiguous index.
 *  @param index Contiguous index.
 *  @param npart Number of parallel processor.
 *  @param ndim Number of dimension of data or model.
 */
MERLIN_EXPORTS Vector<intvec> partition_index_from_contiguous(std::uint64_t index, std::uint64_t npart,
                                                              std::uint64_t ndim);

}  // namespace candy

}  // namespace merlin

#endif  // MERLIN_CANDY_PARTITION_HPP_
