// Copyright 2024 quocdang1998
#ifndef MERLIN_VECTOR_DECLARATION_HPP_
#define MERLIN_VECTOR_DECLARATION_HPP_

namespace merlin::vector {

template <typename T>
class ForwardIterator;
template <typename T>
class ReverseIterator;

template <typename T>
class View;
template <class T, std::uint64_t Capacity>
class StaticVector;
template <class T>
class DynamicVector;

}  // namespace merlin::vector

#endif  // MERLIN_VECTOR_DECLARATION_HPP_
