// Copyright 2024 quocdang1998
#ifndef MERLIN_IO_BYTE_SWAP_HPP_
#define MERLIN_IO_BYTE_SWAP_HPP_

#include <algorithm>    // std::ranges::reverse, std::ranges::transform, std::transform
#include <bit>          // std::endian, std::bit_cast
#include <cstddef>      // std::size_t
#include <cstdint>      // std::uint32_t, std::uint64_t
#include <type_traits>  // std::is_arithmetic_v, std::is_integral_v, std::is_floating_point_v,
                        // std::has_unique_object_representations_v
#include <ranges>       // std::ranges::forward_range

namespace merlin::io {

/** @brief Reverses the bytes in the given integer value.*/
template <std::integral T>
requires std::has_unique_object_representations_v<T>
constexpr T byteswap(T value) noexcept {
    std::array<std::byte, sizeof(T)> representation = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
    std::ranges::reverse(representation);
    return std::bit_cast<T>(representation);
}

/** @brief Convert a numeric type from little endian to native endian, and vice-versa.
 *  @note This function will have no effect on little endian system.
 */
template <typename T>
requires std::is_arithmetic_v<T>
constexpr T little_endian_element(T value) {
    if constexpr (std::endian::native == std::endian::little) {
        return value;
    } else {
        if constexpr (std::is_integral_v<T>) {
            return byteswap(value);
        } else if constexpr (std::is_floating_point_v<T>) {
            if constexpr (sizeof(T) == 4) {
                std::uint32_t int_representation = std::bit_cast<std::uint32_t>(value);
                int_representation = byteswap(int_representation);
                return std::bit_cast<float>(int_representation);
            } else if constexpr (sizeof(T) == 8) {
                auto int_representation = std::bit_cast<std::uint64_t>(value);
                int_representation = byteswap(int_representation);
                return std::bit_cast<double>(int_representation);
            }
        }
    }
    return value;
}

/** @brief Convert an element of numeric type from little endian to native endian, and vice-versa.*/
template <typename T>
requires std::is_arithmetic_v<T>
constexpr void little_endian(T * dest, const T * src, std::size_t count) {
    std::transform(src, src + count, dest, little_endian_element<T>);
}

/** @brief Convert a range of numeric type from little endian to native endian, and vice-versa.
 *  @details This function will have no effect on little endian system.
 */
template <typename R>
requires std::ranges::forward_range<R> && std::is_arithmetic_v<std::ranges::range_value_t<R>>
void little_endian_range(R & range) {
    if constexpr (std::endian::native == std::endian::little) {
        return;
    } else {
        std::ranges::transform(range, range.begin(), little_endian<std::ranges::range_value_t<R>>);
    }
}

}  // namespace merlin::io

#endif  // MERLIN_IO_BYTE_SWAP_HPP_
