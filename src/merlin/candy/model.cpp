// Copyright 2023 quocdang1998
#include "merlin/candy/model.hpp"

#include <cstring>  // std::memcpy
#include <sstream>  // std::ostringstream
#include <utility>  // std::exchange

#include "merlin/logger.hpp"  // FAILURE
#include "merlin/utils.hpp"  // merlin::prod_elements

namespace merlin {

// --------------------------------------------------------------------------------------------------------------------
// Model
// --------------------------------------------------------------------------------------------------------------------

// Constructor from shape and rank
candy::Model::Model(const intvec & shape, std::uint64_t rank) : parameters_(shape.size()), rank_(rank) {
    // check rank
    if (rank == 0) {
        FAILURE(std::invalid_argument, "Cannot initialize a canonical model with rank zero.\n");
    }
    // allocate data
    for (std::uint64_t i_dim = 0; i_dim < shape.size(); i_dim++) {
        parameters_[i_dim] = Vector<double>(shape[i_dim]*rank);
    }
}

// Constructor from model values
candy::Model::Model(const Vector<Vector<double>> & parameter, std::uint64_t rank) : parameters_(parameter),
rank_(rank) {
    // check rank
    if (rank == 0) {
        FAILURE(std::invalid_argument, "Cannot initialize a canonical model with rank zero.\n");
    }
    for (std::uint64_t i_dim = 0; i_dim < parameter.size(); i_dim++) {
        if (parameter[i_dim].size() % rank != 0) {
            FAILURE(std::invalid_argument, "Size of all canonical model vectors must be divisible by the rank.\n");
        }
    }
}

// String representation
std::string candy::Model::str(void) const {
    std::ostringstream out_stream;
    out_stream << "<Candecomp Model(";
    for (std::uint64_t i_dim = 0; i_dim < this->ndim(); i_dim++) {
        out_stream << ((i_dim != 0) ? " " : "");
        out_stream << "<";
        std::uint64_t dim_shape = this->parameters_[i_dim].size() / this->rank_;
        for (std::uint64_t i_index = 0; i_index < dim_shape; i_index++) {
            Vector<double> rank_vector;
            rank_vector.assign(const_cast<double *>(this->parameters_[i_dim].data()) + i_index*this->rank_, this->rank_);
            out_stream << ((i_index != 0) ? " " : "");
            out_stream << rank_vector.str();
        }
        out_stream << ">";
    }
    out_stream << ")>";
    return out_stream.str();
}

// Destructor
candy::Model::~Model(void) {}

}  // namespace merlin
