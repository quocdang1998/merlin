// Copyright 2022 quocdang1998
#include "ap3_mpo/help_message.hpp"

namespace merlin::ext::ap3mpo {

const char * help_message = R"(Retrieve microscopic cross-section from an MPO.
Options:
    Help mode:
        -h, --help: Print help message.
    Query mode: get names of geometries, energy meshes, isotopes and reactions presenting in the MPO.
        -q, --query: Query the MPO
    Get data from MPO:
        -g, --geometry: Name of geometry.
        -e, --energymesh: Name of energy mesh.
        -i, --isotope: Name of isotope.
        -r, --reaction: Name of reaction.
        -o, --output: Name of output file. Default: "output.txt".
        -xs, --xs-type: Type of cross section, choose between "micro", "macro", "zoneflux", "RR" (reaction rate).
                        Default: "micro".
        --no-thread-safe: Multiple threads can read/write the file concurrently.
Result:
    An serialized array of microscopic homogenized cross-section, which can be read with merlin::array::Stock.
)";

}
