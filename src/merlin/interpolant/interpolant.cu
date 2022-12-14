// Copyright 2022 quocdang1998
#include "merlin/interpolant/interpolant.hpp"

#include "merlin/array/parcel.hpp"  // merlin::array::Parcel
#include "merlin/interpolant/cartesian_grid.hpp"  // merlin::interpolant::CartesianGrid

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Calculate coefficients
// --------------------------------------------------------------------------------------------------------------------

// Calculate nlagrange coefficient with GPU parallelism (incomplet)
array::Array calc_lagrange_coeffs_gpu(const interpolant::CartesianGrid * pgrid, const array::Array * pvalue,
                                      const Vector<array::Slice> & slices, const cuda::Stream & stream) {
    // copy grid to GPU
    interpolant::CartesianGrid * gpu_grid;
    ::cudaMalloc(&gpu_grid, pgrid->malloc_size());
    pgrid->copy_to_gpu(gpu_grid, gpu_grid+1);
    // copy array to parcel
    array::Parcel gpu_value(*pvalue, stream);
    array::Parcel * gpu_parcel;
    ::cudaMalloc(&gpu_parcel, gpu_value.malloc_size());
    gpu_value.copy_to_gpu(gpu_parcel, gpu_parcel+1);

    // free data allocated on GPU
    ::cudaFree(gpu_grid);
    ::cudaFree(gpu_parcel);
    return array::Array();
}

}  // namespace merlin
