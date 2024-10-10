
#include "merlin/array/array.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/logger.hpp"

using namespace merlin;

int main(void) {
    // set GPU
    cuda::Device gpu(0);
    gpu.set_as_current();

    // initialize an tensor
    double A_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Index dims = {2, 3};
    Index strides = {5 * sizeof(double), 2 * sizeof(double)};
    array::Array A(A_data, dims, strides, false);

    // copy data to GPU and print each element of the tensor
    cuda::Stream s(cuda::StreamSetting::Default);
    array::Parcel B(A.shape());
    B.transfer_data_to_gpu(A, s);
    Message("First Parcel object: %s\n", B.str().c_str());
    Message("Current GPU ID: %s\n", cuda::Device::get_current_gpu().str().c_str());

    // copy data to the same GPU
    array::Parcel B_(B);
    Message("Parcel object copied to the same GPU: %s\n", B_.str().c_str());
    Message("Current GPU ID: %s\n", cuda::Device::get_current_gpu().str().c_str());

    // copy data to another GPU
    cuda::Device second_gpu(1);
    second_gpu.set_as_current();
    array::Parcel B1(B);
    Message("Parcel object copied to another GPU: %s\n", B1.str().c_str());
    Message("Current GPU ID: %s\n", cuda::Device::get_current_gpu().str().c_str());
}
