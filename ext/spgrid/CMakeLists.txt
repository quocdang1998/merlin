list(APPEND SPARSE_GRID_SRC_CPP
     interpolator.cpp
     sparse_grid.cpp
     utils.cpp
)
list(APPEND SPARSE_GRID_SRC_CU
     interpolator.cu
)
list(APPEND SPARSE_GRID_SRC_RDC
     sparse_grid.rdc
     utils.rdc
)
list(TRANSFORM SPARSE_GRID_SRC_CPP PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/spgrid/)
list(TRANSFORM SPARSE_GRID_SRC_CU  PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/spgrid/)
list(TRANSFORM SPARSE_GRID_SRC_RDC PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/spgrid/)
if (MERLIN_CUDA)
    set_source_files_properties(${SPARSE_GRID_SRC_RDC} PROPERTIES LANGUAGE CUDA)
    list(APPEND SPARSE_GRID_SRC_CPP ${SPARSE_GRID_SRC_CU})
else()
    set_source_files_properties(${SPARSE_GRID_SRC_RDC} PROPERTIES LANGUAGE CXX)
endif(MERLIN_CUDA)

add_library(libspgrid STATIC ${SPARSE_GRID_SRC_CPP} ${SPARSE_GRID_SRC_RDC})
if (MERLIN_CUDA)
    set_property(TARGET libspgrid PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif(MERLIN_CUDA)
set_property(TARGET libspgrid PROPERTY OUTPUT_NAME spgrid)
set_property(TARGET libspgrid PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)
set_property(TARGET libspgrid PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(libspgrid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(libspgrid PUBLIC libmerlin)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
        DESTINATION "include" FILES_MATCHING PATTERN "*.[ht]pp")

add_executable(test_spgrid "test.cpp")
target_include_directories(test_spgrid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(test_spgrid PUBLIC libspgrid)

# add_executable(ap3_mpo ${AP3_MPO_SRC_CPP})
# set_property(TARGET ap3_mpo PROPERTY OUTPUT_NAME ap3_mpo)
# set_property(TARGET ap3_mpo PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
# target_include_directories(ap3_mpo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${HDF5_INCLUDE_DIRS})
# target_link_libraries(ap3_mpo PUBLIC libmerlin ${HDF5_CXX_LIBRARIES} ${HDF5_C_LIBRARIES})
# set_property(TARGET ap3_mpo PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
#
# install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/
#         DESTINATION "include" FILES_MATCHING PATTERN "*.[ht]pp")
# install(TARGETS ap3_mpo RUNTIME DESTINATION "bin")
