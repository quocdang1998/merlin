// Copyright 2022 quocdang1998
#ifndef AP3_XS_HPP_
#define AP3_XS_HPP_

#include <list>  // std::list
#include <string>  // std::string
#include <vector>  // std::vector

#include "H5Cpp.h"  // H5::File

#include "merlin/array/nddata.hpp"  // merlin::array::NdData

#include "ap3_mpo/declaration.hpp"  // merlin::ext::ap3mpo::Ap3HomogXS
#include "ap3_mpo/properties.hpp"  // merlin::ext::ap3mpo::Ap3Geometry, merlin::ext::ap3mpo::Ap3EnergyMesh
                                   // merlin::ext::ap3mpo::Ap3StateParam, merlin::ext::ap3mpo::Ap3Isotope

namespace merlin {

/** @brief Microscopic cross section data read from MPO.*/
class ext::ap3mpo::Ap3HomogXS {
  public:
    Ap3HomogXS(void) = default;
    Ap3HomogXS(const std::string & filename,
               const std::string & geometry_id, const std::string & energy_mesh_id,
               const std::string & isotope, const std::string & reaction);

    Ap3HomogXS(const ext::ap3mpo::Ap3HomogXS & src) = default;
    Ap3HomogXS & operator=(const ext::ap3mpo::Ap3HomogXS & src) = default;
    Ap3HomogXS(ext::ap3mpo::Ap3HomogXS && src) = default;
    Ap3HomogXS & operator=(ext::ap3mpo::Ap3HomogXS && src) = default;

    constexpr ext::ap3mpo::Ap3StateParam & state_param(void) noexcept {return this->state_param_;}
    constexpr const ext::ap3mpo::Ap3StateParam & state_param(void) const noexcept {return this->state_param_;}
    int num_linked_instances(void) {return this->linked_instances_.size();}

    ext::ap3mpo::Ap3HomogXS & operator+=(ext::ap3mpo::Ap3HomogXS & other);
    intvec get_output_shape(void);
    void assign_destination_array(array::NdData & dest);
    void write_to_stock(const ext::ap3mpo::Ap3StateParam & pspace, const std::string & xstype = "micro");

    ~Ap3HomogXS(void) = default;

  protected:
    /** @brief Geometry ID.*/
    ext::ap3mpo::Ap3Geometry geometry_;
    /** @brief Energy mesh.*/
    ext::ap3mpo::Ap3EnergyMesh energymesh_;
    /** @brief Isotope.*/
    ext::ap3mpo::Ap3Isotope isotope_;
    /** @brief Reaction name.*/
    ext::ap3mpo::Ap3Reaction reaction_;

    /** @brief Parameters.*/
    ext::ap3mpo::Ap3StateParam state_param_;
    /** @brief HDF5 file.*/
    H5::H5File mpo_file_;
    /** @brief Output group.*/
    H5::Group output_;
    /** @brief List of linked instances.*/
    std::list<ext::ap3mpo::Ap3HomogXS *> linked_instances_;
    /** @brief Data to be serialized.*/
    array::NdData * pdata_;
};

}  // namespace merlin

#endif  // AP3_XS_HPP_
