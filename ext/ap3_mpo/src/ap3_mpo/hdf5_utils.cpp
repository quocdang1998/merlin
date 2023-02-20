// Copyright 2022 quocdang1998
#include "ap3_mpo/hdf5_utils.hpp"

#include "merlin/logger.hpp"  // FAILURE

namespace merlin {

// Left trim a string
static std::string & ltrim(std::string & s) {
    auto it = std::find_if(s.begin(), s.end(), [](char c) {return !std::isspace<char>(c, std::locale::classic());});
    s.erase(s.begin(), it);
    return s;
}

// Right trim a string
static std::string & rtrim(std::string & s) {
    auto it = std::find_if(s.rbegin(), s.rend(), [](char c) {return !std::isspace<char>(c, std::locale::classic());});
    s.erase(it.base(), s.end());
    return s;
}

// Trim a string.
std::string & ext::ap3mpo::trim(std::string & s) {
    return ltrim(rtrim(s));
}

// Check if a string is in an array (convert to lowercase)
std::uint64_t ext::ap3mpo::check_string_in_array(std::string element, std::vector<std::string> array) {
    for (int i = 0; i < array.size(); i++) {
        if (ext::ap3mpo::lowercase(ext::ap3mpo::trim(array[i])).compare(ext::ap3mpo::lowercase(element)) == 0) {
            return i;
        }
    }
    return UINT64_MAX;
}

// List all subgroups and dataset in a group
std::vector<std::string> ext::ap3mpo::ls_groups(H5::Group * group, const char * substring) {
    // function retrieving the name of subgroups and datasets in a given group
    auto get_name = [] (hid_t loc_id, char const * name, const H5L_info_t * info, void * operator_data) {
        H5O_info_t infobuf;
        #if H5Oget_info_by_name_vers < 2
        H5Oget_info_by_name(loc_id, name, &infobuf, H5P_DEFAULT);
        #else
        H5Oget_info_by_name(loc_id, name, &infobuf, H5O_INFO_BASIC, H5P_DEFAULT);
        #endif  // H5Oget_info_by_name_vers
        auto * ptr_data = static_cast<std::vector<std::string> *>(operator_data);
        std::string element_name(name);
        ptr_data->push_back(std::string(name));
        return 0;
    };
    // iterate over all group
    std::vector<std::string> result;
    H5Literate(group->getId(), H5_INDEX_NAME, H5_ITER_NATIVE, NULL, get_name, (void *) & result);
    // remove if string not contains substring
    std::string s(substring);
    auto check_not_substring = [&s] (const std::string & name) {
        return (!s.empty()) && (name.find(s) == std::string::npos);
    };
    result.erase(std::remove_if(result.begin(), result.end(), check_not_substring), result.end());
    return result;
}

}  // namespace merlin
