#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/launcher.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optmz/adam.hpp"
#include "merlin/candy/optmz/adagrad.hpp"
#include "merlin/candy/optmz/grad_descent.hpp"
#include "merlin/cuda/memory.hpp"
#include "merlin/cuda/stream.hpp"
#include "merlin/logger.hpp"
#include "merlin/vector.hpp"
#include "merlin/utils.hpp"


int main (void) {
    // initialize train data
    double data[6] = {1.2, 2.3, 3.6, 4.8, 7.1, std::nan("")};
    // double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data_cpu(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data_cpu.str().c_str());
    merlin::array::Parcel train_data(train_data_cpu.shape());
    train_data.transfer_data_to_gpu(train_data_cpu);

    // initialize model
    merlin::candy::Model model(train_data.shape(), 2);
    model.initialize(train_data_cpu, merlin::candy::RandomInitializer::NormalDistribution, 24);
    merlin::candy::Model model_ref_cpu = model;

    // initialize optimizer
    // merlin::candy::optmz::GradDescent optimizer(0.5);
    merlin::candy::optmz::AdaGrad optimizer(0.5, model.size());
    // merlin::candy::optmz::Adam optimizer(0.1, 0.9, 0.999, model.size(), 1.0e-8);
    merlin::candy::Optimizer * optimizer_gpu = optimizer.new_gpu();

    // copy model and data to GPU
    merlin::cuda::Memory mem(0, model, train_data);
    merlin::candy::Model * model_gpu = mem.get<0>();
    merlin::array::Parcel * train_data_gpu = mem.get<1>();

    // initialize launcher
    merlin::candy::Launcher launch(model_gpu, train_data_gpu, optimizer_gpu, model.size(), train_data.ndim(),
                                   model.sharedmem_size() + train_data.sharedmem_size() + optimizer.sharedmem_size(),
                                   32);
    merlin::candy::Launcher launch_reference(model_ref_cpu, train_data_cpu, optimizer, 1);

    // launch and wait
    launch.launch_async(5, merlin::candy::TrainMetric::AbsoluteSquare);
    launch_reference.launch_async(5, merlin::candy::TrainMetric::AbsoluteSquare);
    launch.synchronize();
    launch_reference.synchronize();

    // copy model back to CPU and copare with CPU result
    model.copy_from_gpu(model_gpu);
    MESSAGE("Model after trained on GPU: %s\n", model.str().c_str());
    MESSAGE("Model after trained on CPU: %s\n", model_ref_cpu.str().c_str());

    // print evaluation
    for (std::uint64_t i = 0; i < train_data.size(); i++) {
        merlin::intvec index = merlin::contiguous_to_ndim_idx(i, data_dims);
        MESSAGE("Model evaluation at (%s): %f\n", index.str().c_str(), model.eval(index));
    }

    // release memory
    merlin::candy::optmz::GradDescent::delete_gpu(optimizer_gpu);
}
