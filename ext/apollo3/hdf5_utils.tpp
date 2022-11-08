// Copyright 2022 quocdang1998
#ifndef HDF5_UTILS_TPP_
#define HDF5_UTILS_TPP_

#include <cstdint>  // std::uint64_t

template <typename T>
std::pair<std::vector<T>, merlin::intvec> get_dset(H5::Group * group, char const * dset_address) {
    // open dataset
    H5::DataSet dset = group->openDataSet(dset_address);
    // get data shape
    H5::DataSpace dspace = dset.getSpace();
    std::uint64_t ndims = dspace.getSimpleExtentNdims();
    std::vector<hsize_t> shape(ndims);
    dspace.getSimpleExtentDims(shape.data());
    // get size of an element
    std::uint64_t element_size = dset.getDataType().getSize();
    // check size of data if not std::string
    if constexpr (!(std::is_same_v<T, std::string>)) {
        if (element_size != sizeof(T)) {
            FAILURE(std::runtime_error, "Incorrect type provided to the template.\n");
        }
    }
    // read data to a buffer
    std::uint64_t npoint = dspace.getSimpleExtentNpoints();
    std::vector<char> buffer(element_size*npoint);
    dset.read(buffer.data(), dset.getDataType());
    std::vector<T> data(npoint);
    if constexpr (std::is_same_v<T, std::string>) {
        // convert to vector of std::string
        for (int i = 0; i < npoint; i++) {
            data[i].assign(buffer.data() + i*element_size, element_size);
        }
    } else {
        // move buffer to data
        data.data() = reinterpret_cast<T *>(buffer.data());
        buffer.data() = NULL;
    }
    // close dataset
    dset.close();
    // convert shape to intvec
    merlin::intvec data_shape(shape.data(), ndims);
    return std::pair<std::vector<T>, merlin::intvec>(data, data_shape);
}

#endif  // HDF5_UTILS_TPP_
