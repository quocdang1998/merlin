// Copyright 2022 quocdang1998
#include "ap3_mpo/query_mpo.hpp"

#include "H5Cpp.h"  // H5::H5File, H5::Group

#include "ap3_mpo/hdf5_utils.hpp"  // ap3_mpo::ls_group, ap3_mpo::get_dset, ap3_mpo::trim, ap3_mpo::append_suffix

namespace ap3_mpo {

// Print geometries, energy meshes,isotopes, reactions present in MPO
void query_mpo(const std::string & filename) {
    // open file
    H5::H5File mpo_file(filename.c_str(), H5F_ACC_RDONLY);
    H5::Group root = mpo_file.openGroup("/");
    // read geometries
    std::printf("    Geometries ");
    auto [geometry_names, n_geometry] = get_dset<std::string>(&root, "geometry/GEOMETRY_NAME");
    std::printf("[%zu] :", geometry_names.size());
    for (std::string & geometry_name : geometry_names) {
        std::printf(" %s", trim(geometry_name).c_str());
    }
    std::printf("\n");
    // read energy meshes
    std::printf("    Energy meshes ");
    auto [energymesh_names, n_energymesh] = get_dset<std::string>(&root, "energymesh/ENERGYMESH_NAME");
    std::printf("[%zu] :", energymesh_names.size());
    for (int i_emesh = 0; i_emesh < energymesh_names.size(); i_emesh++) {
        std::string & energymesh_name = energymesh_names[i_emesh];
        std::printf(" %s", trim(energymesh_name).c_str());
        std::string emesh_group_name = append_suffix("energymesh/energymesh_", i_emesh);
        H5::Group emesh_group = root.openGroup(emesh_group_name.c_str());
        auto [n_group, _] = get_dset<int>(&emesh_group, "NG");
        std::printf("[%d]", n_group[0]);
    }
    std::printf("\n");
    // read isotopes
    std::printf("    Isotopes ");
    auto [isotope_names, n_isotope] = get_dset<std::string>(&root, "contents/isotopes/ISOTOPENAME");
    std::printf("[%zu] :", isotope_names.size());
    for (std::string & isotope_name : isotope_names) {
        std::printf(" %s", trim(isotope_name).c_str());
    }
    std::printf("\n");
    // read reactions
    std::printf("    Reactions ");
    auto [reaction_names, n_reaction] = get_dset<std::string>(&root, "contents/reactions/REACTIONAME");
    std::printf("[%zu] :", reaction_names.size());
    for (std::string & reaction_name : reaction_names) {
        std::printf(" %s", trim(reaction_name).c_str());
    }
    std::printf("\n");
}

}  // namespace ap3_mpo
