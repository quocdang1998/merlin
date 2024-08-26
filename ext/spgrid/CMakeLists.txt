# source list
list(APPEND SPARSE_GRID_SRC_CPP hier_interpolator.cpp sparse_grid.cpp utils.cpp)
list(TRANSFORM SPARSE_GRID_SRC_CPP PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/spgrid/)

# add library
add_library(libspgrid STATIC ${SPARSE_GRID_SRC_CPP})
if(MERLIN_CUDA)
    set_property(TARGET libspgrid PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif(MERLIN_CUDA)
set_property(TARGET libspgrid PROPERTY OUTPUT_NAME spgrid)
set_property(TARGET libspgrid PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
set_property(TARGET libspgrid PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(libspgrid PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                                            $<INSTALL_INTERFACE:include>)
target_link_libraries(libspgrid PUBLIC libmerlin)

# add install file
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ DESTINATION "include" FILES_MATCHING PATTERN "*.[ht]pp")
install(TARGETS libspgrid RUNTIME DESTINATION "lib")
install(TARGETS libspgrid EXPORT libspgrid-config LIBRARY DESTINATION "lib" ARCHIVE DESTINATION "lib"
        RUNTIME DESTINATION "bin")
install(EXPORT libspgrid-config DESTINATION "lib/cmake" FILE libspgridConfig.cmake NAMESPACE spgrid::)

# add test
add_executable(test_spgrid "test.cpp")
target_include_directories(test_spgrid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_spgrid PUBLIC libspgrid)
