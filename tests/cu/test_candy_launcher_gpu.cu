#include "merlin/array/array.hpp"
#include "merlin/array/operation.hpp"
#include "merlin/array/parcel.hpp"
#include "merlin/candy/model.hpp"
#include "merlin/candy/optimizer.hpp"
#include "merlin/candy/optmz/adagrad.hpp"
#include "merlin/candy/optmz/adam.hpp"
#include "merlin/candy/optmz/grad_descent.hpp"
#include "merlin/candy/train/cpu_trainer.hpp"
#include "merlin/candy/train/gpu_trainer.hpp"
#include "merlin/candy/trainer.hpp"
#include "merlin/candy/trial_policy.hpp"
#include "merlin/cuda/device.hpp"
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
    Index data_dims = {2, 3}, data_strides = {data_dims[1] * sizeof(double), sizeof(double)};
    array::Array train_data(data, data_dims, data_strides);
    Message("Data: {}\n", train_data.str());

    // copy data to GPU
    array::Parcel gpu_data(train_data.shape());
    gpu_data.transfer_data_to_gpu(train_data);

    // initialize model
    candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    Message("Model before trained: {}\n", model.str());

    // initialize optimizer
    // candy::Optimizer opt = candy::create_grad_descent(0.2);
    candy::Optimizer opt = candy::optmz::create_adagrad(0.1, model.num_params());

    // initialize gpu trainer
    // Synchronizer gpu_sync(cuda::Stream{});
    Synchronizer gpu_sync(ProcessorType::Gpu);
    candy::train::GpuTrainer gpu_trainer(2, gpu_sync);
    gpu_trainer.set_model("foo", model);
    gpu_trainer.set_optmz("foo", opt);
    gpu_trainer.set_data("foo", gpu_data);
    gpu_sync.synchronize();

    // initialize cpu trainer
    Synchronizer cpu_sync(ProcessorType::Cpu);
    candy::train::CpuTrainer cpu_trainer(2, cpu_sync);
    cpu_trainer.set_model("foo", model);
    cpu_trainer.set_optmz("foo", opt);
    cpu_trainer.set_data("foo", train_data);
    cpu_sync.synchronize();

    // dry-run on GPU
    candy::TrialPolicy policy(1, 9, 90);
    DoubleVec error(policy.sum());
    std::uint64_t count;
    std::map<std::string, std::pair<double *, std::uint64_t *>> tracking_map;
    tracking_map["foo"] = {error.data(), &count};
    gpu_trainer.dry_run(tracking_map, policy);
    gpu_sync.synchronize();
    if (count != policy.sum()) {
        Fatal<std::runtime_error>("Invalid optimizer.\n");
    }
    Message("Errors: ") << error.str() << "\n";

    /*cpu_trainer.update_until(1000, 0.01);
    gpu_trainer.update_until(1000, 0.01);*/
    cpu_trainer.update_for(100);
    gpu_trainer.update_for(100);
    cpu_sync.synchronize();
    gpu_sync.synchronize();

    // print model
    candy::Model gpu_model = gpu_trainer.get_model("foo");
    Message("Model after trained (GPU): {}\n", gpu_model.str());
    candy::Model cpu_model = cpu_trainer.get_model("foo");
    Message("Model after trained (CPU): {}\n", cpu_model.str());

    // print reconstructed data
    array::Array cpu_recdata(train_data.shape());
    std::map<std::string, array::Array *> cpu_recmap;
    cpu_recmap["foo"] = &cpu_recdata;
    cpu_trainer.reconstruct(cpu_recmap);
    array::Parcel gpu_recdata(train_data.shape());
    std::map<std::string, array::Parcel *> gpu_recmap;
    gpu_recmap["foo"] = &gpu_recdata;
    gpu_trainer.reconstruct(gpu_recmap);
    cpu_sync.synchronize();
    gpu_sync.synchronize();
    Message("Reconstructed data (GPU): {}\n", gpu_recdata.str());
    Message("Reconstructed data (CPU): {}\n", cpu_recdata.str());

    // print error
    double cpu_rmse, cpu_rmae;
    std::map<std::string, std::array<double *, 2>> cpu_errmap;
    cpu_errmap["foo"] = {&cpu_rmse, &cpu_rmae};
    cpu_trainer.get_error(cpu_errmap);
    double gpu_rmse, gpu_rmae;
    std::map<std::string, std::array<double *, 2>> gpu_errmap;
    gpu_errmap["foo"] = {&gpu_rmse, &gpu_rmae};
    gpu_trainer.get_error(gpu_errmap);
    cpu_sync.synchronize();
    gpu_sync.synchronize();
    Message("Error (GPU): {} {}\n", gpu_rmse, gpu_rmae);
    Message("Error (CPU): {} {}\n", cpu_rmse, cpu_rmae);

    // export models
    gpu_trainer.export_models();
}
