find_package(HDF5 COMPONENTS CXX)
list(APPEND AP3_MPO_SRC_CPP
     ap3_xs.cpp
     hdf5_utils.cpp
     help_message.cpp
     glob.cpp
     main.cpp
     properties.cpp
     query_mpo.cpp
)
list(TRANSFORM AP3_MPO_SRC_CPP PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/src/ap3_mpo/)

add_executable(ap3_mpo ${AP3_MPO_SRC_CPP})
set_property(TARGET ap3_mpo PROPERTY OUTPUT_NAME ap3_mpo)
set_property(TARGET ap3_mpo PROPERTY CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(ap3_mpo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${HDF5_INCLUDE_DIRS})
target_link_libraries(ap3_mpo libmerlin ${HDF5_CXX_LIBRARIES} ${HDF5_C_LIBRARIES})
