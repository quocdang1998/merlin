// Copyright 2024 quocdang1998
#ifndef MERLIN_VECTOR_VIEW_TPP_
#define MERLIN_VECTOR_VIEW_TPP_

#include <ostream>  // std::ostream
#include <sstream>  // std::ostringstream

namespace merlin {

// ---------------------------------------------------------------------------------------------------------------------
// Concepts
// ---------------------------------------------------------------------------------------------------------------------

namespace vector {

// Type can be formatted to an std::ostream
template <typename T>
concept Streamable = requires(std::ostream & os, const T & obj) {
    { os << obj } -> std::same_as<std::ostream &>;
};

// Type having a method named ``str(void)``` to become string
template <typename T>
concept Representable = requires(const T & obj) {
    { obj.str() } -> std::convertible_to<std::string>;
};

}  // namespace vector

// ---------------------------------------------------------------------------------------------------------------------
// Range view
// ---------------------------------------------------------------------------------------------------------------------

// String representation
template <typename T>
std::string vector::View<T>::str(const char * sep) const {
    static_assert(vector::Streamable<T> || vector::Representable<T>, "Desired type is not representable.\n");
    std::ostringstream os;
    os << "<";
    for (std::uint64_t i = 0; i < this->size_; i++) {
        if constexpr (vector::Streamable<T>) {
            os << this->data_[i];
        } else if constexpr (vector::Representable<T>) {
            os << this->data_[i].str();
        }
        if (i != this->size_ - 1) {
            os << sep;
        }
    }
    os << ">";
    return os.str();
}

}  // namespace merlin

#endif  // MERLIN_VECTOR_VIEW_TPP_
