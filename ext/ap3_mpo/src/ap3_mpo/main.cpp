// Copyright 2022 quocdang1998
#include <iterator>  // std::make_move_iterator
#include <string>  // std::string
#include <vector>  // std::vector

#include "merlin/array/stock.hpp"  // merlin::array::Stock
#include "merlin/logger.hpp"  // MESSAGE

#include "ap3_mpo/ap3_xs.hpp"  // merlin::ext::ap3mpo::Ap3HomogXS
#include "ap3_mpo/help_message.hpp"  // merlin::ext::ap3mpo::help_message
#include "ap3_mpo/glob.hpp"  // merlin::ext::ap3mpo::glob
#include "ap3_mpo/query_mpo.hpp"  // merlin::ext::ap3mpo::query_mpo

int main(int argc, char * argv[]) {
    // parse argument
    unsigned int mode = 0;
    std::string geometry, energymesh, isotope, reaction, output = "output.txt", xstype = "micro";
    bool thread_safe = true;
    std::vector<std::string> filenames;
    for (int i = 1; i < argc; i++) {
        std::string argument(argv[i]);
        if (!argument.compare("-h") || !argument.compare("--help")) {
            mode |= 1;
        } else if (!argument.compare("-q") || !argument.compare("--query")) {
            mode |= 2;
        } else if (!argument.compare("-i") || !argument.compare("--isotope")) {
            isotope = std::string(argv[++i]);
            mode |= 4;
        } else if (!argument.compare("-r") || !argument.compare("--reaction")) {
            reaction = std::string(argv[++i]);
            mode |= 4;
        } else if (!argument.compare("-e") || !argument.compare("--energy-mesh")) {
            energymesh = std::string(argv[++i]);
            mode |= 4;
        } else if (!argument.compare("-g") || !argument.compare("--geometry")) {
            geometry = std::string(argv[++i]);
            mode |= 4;
        } else if (!argument.compare("-o") || !argument.compare("--output")) {
            output = std::string(argv[++i]);
            mode |= 4;
        } else if (!argument.compare("-xs") || !argument.compare("--xs-type")) {
            xstype = std::string(argv[++i]);
            mode |= 4;
        } else if (!argument.compare("--no-thread-safe")) {
            thread_safe = false;
            mode |= 4;
        } else {
            std::vector<std::string> glob_expanded = merlin::ext::ap3mpo::glob(argument);
            filenames.insert(filenames.end(), std::make_move_iterator(glob_expanded.begin()),
                             std::make_move_iterator(glob_expanded.end()));
        }
    }
    // help mode
    if (mode == 1) {
        std::printf("%s", merlin::ext::ap3mpo::help_message);
        return 0;
    }
    // query mode
    if (mode == 2) {
        for (const std::string & filename : filenames) {
            MESSAGE("Querying file %s...\n", filename.c_str());
            merlin::ext::ap3mpo::query_mpo(filename);
        }
        return 0;
    }
    // write to file mode
    if (mode == 4) {
        // build combined object
        merlin::ext::ap3mpo::Ap3HomogXS combined_mpo;
        std::vector<merlin::ext::ap3mpo::Ap3HomogXS *> component_ptr;
        for (int i = 0; i < filenames.size(); i++) {
            const std::string & filename = filenames[i];
            MESSAGE("Reading MPO file \"%s\"...\n", filename.c_str());
            auto mpofile = new merlin::ext::ap3mpo::Ap3HomogXS(filename, geometry, energymesh, isotope, reaction);
            component_ptr.push_back(mpofile);
            combined_mpo += *component_ptr[i];
        }
        // write data to file
        merlin::array::Stock stock(output, combined_mpo.get_output_shape(), 0, thread_safe);
        combined_mpo.assign_destination_array(stock);
        combined_mpo.write_to_stock(combined_mpo.state_param(), xstype);
        // deallocate memory
        for (int i = 0; i < component_ptr.size(); i++) {
            delete component_ptr[i];
        }
        return 0;
    }
    // argument not match together
    FAILURE(std::runtime_error, "Argument options not match. Execute \"ap3_mpo --help\" for more information.\n");
    return 1;
}
