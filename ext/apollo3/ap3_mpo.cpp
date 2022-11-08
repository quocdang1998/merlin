// Copyright 2022 quocdang1998
#include "ap3_mpo.hpp"

#include <cstdlib>  // std::atoi

#include "H5Cpp.h"  // H5::Group, H5::H5File, H5F_ACC_RDONLY
#include "merlin/logger.hpp"  // MESSAGE, FAILURE

#include "hdf5_utils.hpp"  // ls_groups

Ap3HomogCrossSection::Ap3HomogCrossSection(const std::string & filename,
                                           const std::string & isotope, const std::string & reaction,
                                           unsigned int energy_group) {
    // read metadata
    this->isotope_ = isotope;
    this->reaction_ = reaction;
    this->energy_group_ = energy_group;
    // open file
    H5::H5File mpofile(filename.c_str(), H5F_ACC_RDONLY);
    H5::Group root = mpofile.openGroup("/");

    std::vector<std::string> groups = ls_groups(&root);
    for (int i = 0; i < groups.size(); i++) {
        MESSAGE("%s.\n", groups[i].c_str());
    }

    // close file
    root.close();
    mpofile.close();
}

