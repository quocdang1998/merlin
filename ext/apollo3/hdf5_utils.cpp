// Copyright 2022 quocdang1998
#include "hdf5_utils.hpp"

#include "merlin/logger.hpp"  // FAILURE

// check if a string is in an array (convert to lowercase)
bool check_string_in_array(std::string element, merlin::Vector<std::string> array) {
    for (const std::string & s : array) {
        if (lowercase(s).find(lowercase(element)) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// list all subgroups and dataset in a group
std::vector<std::string> ls_groups(H5::Group * group) {
    // function retrieving the name of subgroups and datasets in a given group
    auto get_name = [] (hid_t loc_id, char const * name, const H5L_info_t * info, void * operator_data) {
        H5O_info_t infobuf;
        H5Oget_info_by_name (loc_id, name, &infobuf, H5P_DEFAULT);
        auto * ptr_data = static_cast<std::vector<std::string> *>(operator_data);
        ptr_data->push_back(std::string(name));
        return 0;
    };
    // iterate over all group
    std::vector<std::string> result;
    H5Literate(group->getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, get_name, (void *) & result);
    return result;
}

// get 1D string data set
/*
merlin::Vector<std::string> get_string_dset(H5::Group * group, char const * dset_address) {
    // open dataset
    H5::DataSet dset = group->openDataSet(dset_address);
    // get data size
    H5::DataSpace dspace = dset.getSpace();
    hsize_t data_length;
    int data_ndims = dspace.getSimpleExtentDims(&data_length);
    if (data_ndims != 1) {
        FAILURE(std::range_error, "Dataset %s is not a 1D array.\n", dset_address);
    }
    // check data type
    H5T_class_t data_type = dset.getTypeClass();
    if (data_type != H5T_class_t::H5T_STRING) {
        FAILURE(std::runtime_error, "Dataset is not a 1D array of STRING.\n");
    }
    // get string length
    std::uint64_t string_length = dset.getDataType().getSize();
    // read data to a buffer
    char * buffer = new char[string_length * data_length];
    dset.read(buffer, dset.getDataType());
    // copy from buffer to result
    merlin::Vector<std::string> data_read(data_length, std::string(""));
    for (int idx = 0; idx < data_read.size(); idx++) {
        data_read[idx].assign(buffer + idx*string_length, string_length);
    }
    delete[] buffer;
    // close dataset
    dset.close();
    return data_read;
}
*/
