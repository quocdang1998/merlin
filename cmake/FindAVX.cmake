# ======================================================================================================================
# Detect architecture
# ======================================================================================================================

# copy file to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindAVX
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles)

# configure, build and execute avx_detector
if (NOT MERLIN_VECTORIZATION)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_GENERATOR=${CMAKE_GENERATOR} .
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindAVX
        OUTPUT_QUIET
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build . --target avx_detector
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindAVX
        OUTPUT_QUIET
    )
    execute_process(
        COMMAND ./avx_detector
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/FindAVX
        OUTPUT_VARIABLE VECTORIZE_OPTIONS
    )
endif()
string(REPLACE " " ";" MERLIN_VECTORIZATION ${VECTORIZE_OPTIONS})
message(STATUS "Detected CPU architecture ${MERLIN_VECTORIZATION}")
