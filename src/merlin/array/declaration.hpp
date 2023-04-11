// Copyright 2022 quocdang1998
#ifndef MERLIN_ARRAY_DECLARATION_HPP_
#define MERLIN_ARRAY_DECLARATION_HPP_

namespace merlin::array {
class Slice;   // Array slice
class NdData;  // Basic ndim array
class Array;   // CPU Array, defined in array.hpp
class Parcel;  // GPU Array, defined in parcel.hpp
class Stock;   // Out of core array, defined in stock.hpp
}  // namespace merlin::array

#endif  // MERLIN_ARRAY_DECLARATION_HPP_
