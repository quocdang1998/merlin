
#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/logger.hpp"

int main(void) {
    // initialize an tensor
    double A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    merlin::intvec dims = {2, 3};
    merlin::intvec strides = {5*sizeof(double), 2*sizeof(double)};
    merlin::array::Array A(A_data, dims, strides, false);

    // copy data to GPU and print each element of the tensor
    merlin::cuda::Stream s(merlin::cuda::StreamSetting::Default);
    merlin::array::Parcel B(A.shape());
    B.transfer_data_to_gpu(A, s);
    MESSAGE("First Parcel object: %s\n", B.str().c_str());
    MESSAGE("Current GPU ID: %s\n", merlin::cuda::Device::get_current_gpu().str().c_str());

    // copy data to the same GPU
    merlin::array::Parcel B_(B);
    MESSAGE("Parcel object copied to the same GPU: %s\n", B_.str().c_str());
    MESSAGE("Current GPU ID: %s\n", merlin::cuda::Device::get_current_gpu().str().c_str());

    // copy data to another GPU
    merlin::cuda::Device second_gpu(1);
    second_gpu.set_as_current();
    merlin::array::Parcel B1(B);
    MESSAGE("Parcel object copied to another GPU: %s\n", B1.str().c_str());
    MESSAGE("Current GPU ID: %s\n", merlin::cuda::Device::get_current_gpu().str().c_str());
}
