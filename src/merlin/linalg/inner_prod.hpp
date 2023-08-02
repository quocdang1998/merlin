// Copyright 2022 quocdang1998
#ifndef MERLIN_LINALG_INNER_PROD_HPP_
#define MERLIN_LINALG_INNER_PROD_HPP_

#include "merlin/cuda_interface.hpp"  // __cuhostdev__
#include "merlin/vector.hpp"          // merlin::Vector

namespace merlin::linalg {

/** @brief Inner product between 2 vectors.
 *  @details Inner product between 2 vectors @f$ \boldsymbol{a} @f$ and @f$ \boldsymbol{b} @f$:
 *
 *  @f[ \langle \boldsymbol{a}, \boldsymbol{b} \rangle = \boldsymbol{a}^\intercal \boldsymbol{b} = \sum_i a_i b_i @f]
 */
template <typename T>
__cuhostdev__ T inner_product(const Vector<T> & a, const Vector<T> & b) noexcept;

/** @brief Calculate norm of a vector.
 *  @details Norm of a vector @f$ \boldsymbol{a} @f$:
 *
 *  @f[ \lVert \boldsymbol{a} \rVert = \sqrt{\langle \boldsymbol{a}, \boldsymbol{a} \rangle} = \sqrt{\sum_i a_i a_i}
 *  @f]
 */
template <typename T>
__cuhostdev__ T norm(const Vector<T> & a) noexcept;

/** @brief Normalize a vector.
 *  @details Divide entries of a vector @f$ \boldsymbol{a} @f$ by its norm @f$ \lVert \boldsymbol{a} \rVert @f$.
 *
 *  @f[ \boldsymbol{a} \leftarrow \frac{\boldsymbol{a}}{\lVert \boldsymbol{a} \rVert} @f]
 */
template <typename T>
__cuhostdev__ void normalize(Vector<T> & a) noexcept;

}  // namespace merlin::linalg

#include "merlin/linalg/inner_prod.tpp"

#endif  // MERLIN_LINALG_INNER_PROD_HPP_
