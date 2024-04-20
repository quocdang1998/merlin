find_package(OpenMP QUIET REQUIRED)
if(MSVC) # patch the MSVC OpenMP default 2.0 version
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:-openmp:llvm>)
endif(MSVC)
