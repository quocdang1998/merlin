// Copyright 2022 quocdang1998
#ifndef MERLIN_SPLINT_INTERPOLATOR_HPP_
#define MERLIN_SPLINT_INTERPOLATOR_HPP_

#include "merlin/array/nddata.hpp"        // merlin::array::NdData, merlin::array::Array
#include "merlin/cuda/stream.hpp"         // merlin::cuda::Stream
#include "merlin/exports.hpp"             // MERLIN_EXPORTS
#include "merlin/splint/declaration.hpp"  // merlin::splint::CartesianGrid, merlin::splint::Interpolant
#include "merlin/splint/tools.hpp"        // merlin::splint::Method
#include "merlin/vector.hpp"              // merlin::Vector

namespace merlin {

/** @brief Interpolate on a multi-dimensional object.
 *  @details A multi-dimensional interpolation is a linear combination of basis functions. Each basis function is a
 *  product of mono-variable functions.
 *
 *  Thanks to this property, the linear system of equations to solve for the interpolation coefficients can be
 *  decomposed as Kronecker product of one-dimensional matrices, allowing the reduction per dimension of the original
 *  matrix. Furthermore, the process of solving each component matrix can be parallelized over each linear sub-system
 *  created, thus reducing the calculation time of the coefficients.
 */
class splint::Interpolator {
  public:
    /// @name Constructor
    /// @{
    /** @brief Default constructor.*/
    Interpolator(void) = default;
    /** @brief Construct from a CPU array.
     *  @param grid Cartesian grid to interpolate.
     *  @param values C-contiguous array containing the data.
     *  @param method Interpolation method to use for each dimension.
     *  @param num_threads Number of CPU threads to calculate the interpolation.
    */
    MERLIN_EXPORTS Interpolator(const splint::CartesianGrid & grid, array::Array & values,
                                const Vector<splint::Method> & method, std::uint64_t num_threads = 1);
    /** @brief Construct from a GPU array.
     *  @param grid Cartesian grid to interpolate.
     *  @param values C-contiguous array containing the data on GPU.
     *  @param method Interpolation method to use for each dimension.
     *  @param stream CUDA Stream on which the calculation is launched.
     *  @param num_threads Number of CPU threads to calculate the interpolation.
     */
    MERLIN_EXPORTS Interpolator(const splint::CartesianGrid & grid, array::Parcel & values,
                                const Vector<splint::Method> & method, const cuda::Stream & stream = cuda::Stream(),
                                std::uint64_t num_threads = 32);
    /// @}

    /// @name Calculate coefficient
    /// @{
    /** @brief Calculate interpolation coefficients.*/
    void construct_coefficients(void);
    /// @}

    /// @name Destructor
    /// @{
    /** @brief Destructor.*/
    MERLIN_EXPORTS ~Interpolator(void);
    /// @}

  private:
    /** @brief Cartesian grid to interpolate.*/
    splint::CartesianGrid * p_grid_ = nullptr;
    /** @brief Pointer to coefficient array.*/
    array::NdData * p_coeff_ = nullptr;
    /** @brief Interpolation method to applied on each dimension.*/
    Vector<splint::Method> * p_method_ = nullptr;
    /** @brief Pointer to the CUDA Stream used for memory allocation and de-allocation.*/
    std::uintptr_t allocation_stream_;
    /** @brief GPU ID on which the memory are allocated.*/
    unsigned int gpu_id_ = static_cast<unsigned int>(-1);
};

}  // namespace merlin

#endif  // MERLIN_SPLINT_INTERPOLATOR_HPP_
