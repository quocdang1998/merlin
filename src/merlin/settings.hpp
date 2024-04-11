#ifndef MERLIN_SETTINGS_HPP_
#define MERLIN_SETTINGS_HPP_

#include <array>    // std::array
#include <cstdint>  // std::uint64_t
#include <initializer_list>  // std::initializer_list

namespace merlin {

/** @brief Max number of dimensions.*/
inline constexpr const std::uint64_t max_dim = 16;

/** @brief Array of 8 bytes unsigned int.*/
using Index = std::array<std::uint64_t, max_dim>;

/** @brief Array of floatting points.*/
using Point = std::array<double, max_dim>;

/** @brief Array of pointers to floating points.*/
using DPtrArray = std::array<double *, max_dim>;

/** @brief Convertible from pointer.*/
template<class T, class ForwardIterator>
concept ConvertibleFromIterator = requires(ForwardIterator it) {
    {*it} -> std::convertible_to<T>;
};

/** @brief Make an array from another container.*/
template <class T, class ForwardIterator>
requires ConvertibleFromIterator<T, ForwardIterator>
std::array<T, max_dim> make_array(ForwardIterator begin, ForwardIterator end) {
    std::array<T, max_dim> result_array;
    result_array.fill(T());
    typename std::array<T, max_dim>::iterator it_arr = result_array.begin();
    for (ForwardIterator it = begin; it != end; ++it) {
        *(it_arr++) = *(it);
    }
    return result_array;
}


/** @brief Make an array from incomplet initializer list.*/
template <class T>
std::array<T, max_dim> make_array(std::initializer_list<T> list) {
    return make_array<T>(list.begin(), list.end());
}

}  // namespace merlin

#endif  // MERLIN_SETTINGS_HPP_
