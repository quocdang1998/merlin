// Copyright 2022 quocdang1998
#include "ap3_mpo.hpp"  // Ap3HomogCrossSection

int main(int argc, char * argv[]) {
    // parse argument
    std::string filename, isotope, reaction;
    unsigned int energy_group;
    for (int i = 1; i < argc; i++) {
        std::string argument(argv[i]);
        if (!argument.compare("-i") || !argument.compare("--isotope")) {
            isotope = std::string(argv[++i]);
        }
        else if (!argument.compare("-r") || !argument.compare("--reaction")) {
            reaction = std::string(argv[++i]);
        }
        else if (!argument.compare("-e") || !argument.compare("--energy-group")) {
            energy_group = std::atoi(argv[++i]);
        }
        else {
            filename = argument;
        }
    }
    // build Ap3HomogCrossSection object
    Ap3HomogXS homog_xs(filename, isotope, reaction, energy_group);
}
