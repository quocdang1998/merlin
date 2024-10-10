// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_DECLARATION_HPP_
#define MERLIN_SPLINT_DECLARATION_HPP_

namespace merlin::splint {

class Interpolator;

/** @brief Interpolation method.*/
enum class Method : std::uint64_t {
    /** @brief Linear interpolation.*/
    Linear = 0x00,
    /** @brief Polynomial interpolation by Lagrange method.*/
    Lagrange = 0x01,
    /** @brief Polynomial interpolation by Newton method.*/
    Newton = 0x02
};

}  // namespace merlin::splint

#endif  // MERLIN_SPLINT_DECLARATION_HPP_
