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


int main (void) {
    // initialize train data
    double data[6] = {1.2, 2.3, 3.6, 4.8, 7.1, 2.5};
    // double data[6] = {2.5, 3.0, 3.5, 4.45, 5.34, 6.07};
    merlin::intvec data_dims = {2, 3}, data_strides = merlin::array::contiguous_strides(data_dims, sizeof(double));
    merlin::array::Array train_data(data, data_dims, data_strides);
    MESSAGE("Data: %s\n", train_data.str().c_str());

    // initialize model
    merlin::candy::Model model({{1.0, 0.5, 2.1, 0.25}, {2.0, 1.0, 2.4, 1.2, 2.7, 1.6}}, 2);
    MESSAGE("Model before trained: %s\n", model.str().c_str());

    // initialize optimizer
    merlin::candy::Optimizer opt = merlin::candy::create_grad_descent(0.1);

    // create trainer
    merlin::candy::Trainer train(model, std::move(train_data), opt, merlin::ProcessorType::Gpu);
    train.update(10, 1e-1, 5, merlin::candy::TrainMetric::RelativeSquare);
    train.synchronize();

    // copy back to CPU
    merlin::candy::Model trained_model = train.get_model();
    MESSAGE("Model eval: %f\n", trained_model.eval({1,1}));
}
