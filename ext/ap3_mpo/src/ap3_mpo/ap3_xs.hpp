// Copyright 2022 quocdang1998
#ifndef AP3_MPO_AP3_XS_HPP_
#define AP3_MPO_AP3_XS_HPP_

#include <list>    // std::list
#include <string>  // std::string
#include <vector>  // std::vector

#include "H5Cpp.h"  // H5::File

#include "merlin/array/nddata.hpp"  // merlin::array::NdData

#include "ap3_mpo/declaration.hpp"  // ap3_mpo::Ap3HomogXS
#include "ap3_mpo/properties.hpp"   // ap3_mpo::Ap3Geometry, ap3_mpo::Ap3EnergyMesh
                                    // ap3_mpo::Ap3StateParam, ap3_mpo::Ap3Isotope

namespace ap3_mpo {

/** @brief Microscopic cross section data read from MPO.*/
class Ap3HomogXS {
  public:
    Ap3HomogXS(void) = default;
    Ap3HomogXS(const std::string & filename, const std::string & geometry_id, const std::string & energy_mesh_id,
               const std::string & isotope, const std::string & reaction, bool verbose = false);

    Ap3HomogXS(const Ap3HomogXS & src) = default;
    Ap3HomogXS & operator=(const Ap3HomogXS & src) = default;
    Ap3HomogXS(Ap3HomogXS && src) = default;
    Ap3HomogXS & operator=(Ap3HomogXS && src) = default;

    constexpr Ap3StateParam & state_param(void) noexcept { return this->state_param_; }
    constexpr const Ap3StateParam & state_param(void) const noexcept { return this->state_param_; }
    int num_linked_instances(void) { return this->linked_instances_.size(); }

    Ap3HomogXS & operator+=(Ap3HomogXS & other);
    merlin::intvec get_output_shape(void);
    void assign_destination_array(merlin::array::NdData & dest);
    void write_to_stock(const Ap3StateParam & pspace, const std::string & xstype = "micro");

    ~Ap3HomogXS(void) = default;

    /** @brief Verbosity.*/
    bool verbose = false;

  protected:
    /** @brief Geometry ID.*/
    Ap3Geometry geometry_;
    /** @brief Energy mesh.*/
    Ap3EnergyMesh energymesh_;
    /** @brief Isotope.*/
    Ap3Isotope isotope_;
    /** @brief Reaction name.*/
    Ap3Reaction reaction_;

    /** @brief Parameters.*/
    Ap3StateParam state_param_;
    /** @brief HDF5 file.*/
    H5::H5File mpo_file_;
    /** @brief Output group.*/
    H5::Group output_;
    /** @brief List of linked instances.*/
    std::list<Ap3HomogXS *> linked_instances_;
    /** @brief Data to be serialized.*/
    merlin::array::NdData * pdata_;
};

}  // namespace ap3_mpo

#endif  // AP3_MPO_AP3_XS_HPP_
