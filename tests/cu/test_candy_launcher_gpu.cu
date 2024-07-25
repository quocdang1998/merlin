#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/trainer.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optmz/adam.hpp"
#include "merlin/candy/optmz/adagrad.hpp"
#include "merlin/candy/optmz/grad_descent.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/cuda/device.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/logger.hpp"
#include "merlin/synchronizer.hpp"
#include "merlin/utils.hpp"
#include "merlin/vector.hpp"

using namespace merlin;

int main (void) {
    // set GPU
    cuda::Device gpu(0);
    gpu.set_as_current();

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
    // candy::Optimizer opt = candy::create_grad_descent(0.2);
    candy::Optimizer opt = candy::create_rmsprop(0.3, 1e-5, model);

    // create trainer
    std::uint64_t rep = 10;
    double threshold = 1e-5;
    Synchronizer gpu_sync(ProcessorType::Gpu);
    candy::Trainer train_gpu(model, opt, gpu_sync);
    Synchronizer cpu_sync(ProcessorType::Cpu);
    candy::Trainer train_cpu(model, opt, cpu_sync);

    // GPU dryrun
    candy::TrialPolicy policy(1, 9, 40);
    DoubleVec error_by_step(policy.sum());
    std::uint64_t real_iter;
    train_gpu.dry_run(gpu_data, error_by_step, real_iter, policy);
    gpu_sync.synchronize();
    Message("Error dry-run: %s\n", error_by_step.str().c_str());
    bool test = (real_iter == policy.sum());
    if (!test) {
        Fatal<std::runtime_error>("Provided optimizer not compatible with the model.\n");
    }

    // launch update
    train_gpu.set_fname("result_gpu.txt");
    train_gpu.update(gpu_data, rep, threshold, 16, candy::TrainMetric::RelativeSquare, true);
    train_cpu.set_fname("result_cpu.txt");
    train_cpu.update(train_data, rep, threshold, 3, candy::TrainMetric::RelativeSquare, true);

    // synchronize
    gpu_sync.synchronize();
    cpu_sync.synchronize();

    // copy back to CPU
    Message("Model after trained (GPU): %s\n", train_gpu.model().str().c_str());
    Message("Model after trained (CPU): %s\n", train_cpu.model().str().c_str());
}
