#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/trainer.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optmz/adam.hpp"
#include "merlin/candy/optmz/adagrad.hpp"
#include "merlin/candy/optmz/grad_descent.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"
#include "merlin/utils.hpp"

using namespace merlin;

int main (void) {
    // initialize train data
    double data[6] = {1.2, 2.3, 3.6, 4.8, 7.1, 2.5};
    // double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    UIntVec data_dims = {2, 3}, data_strides = {data_dims[1] * sizeof(double), sizeof(double)};
    array::Array train_data(data, data_dims, data_strides);
    Message("Data: %s\n", train_data.str().c_str());

    // copy data to GPU
    array::Parcel gpu_data(train_data.shape());
    gpu_data.transfer_data_to_gpu(train_data);

    // initialize model
    candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    Message("Model before trained: %s\n", model.str().c_str());

    // initialize optimizer
    candy::Optimizer opt = candy::create_grad_descent(0.1);

    // create trainer
    std::uint64_t rep = 10;
    double threshold = 1e-5;
    candy::Trainer train_gpu(model, opt, ProcessorType::Gpu);
    train_gpu.update_gpu(gpu_data, rep, threshold, 16, candy::TrainMetric::RelativeSquare);
    candy::Trainer train_cpu(model, opt, ProcessorType::Cpu);
    train_cpu.update_cpu(train_data, rep, threshold, 3, candy::TrainMetric::RelativeSquare);

    // synchronize
    train_gpu.synchronize();
    train_cpu.synchronize();

    // copy back to CPU
    Message("Model after trained (GPU): %s\n", train_gpu.model().str().c_str());
    Message("Model after trained (CPU): %s\n", train_cpu.model().str().c_str());
}
