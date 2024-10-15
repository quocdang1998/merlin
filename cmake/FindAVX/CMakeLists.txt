cmake_minimum_required(VERSION 3.20)

project(find_avx LANGUAGES CXX)

# add executable
add_executable(avx_detector "${CMAKE_CURRENT_SOURCE_DIR}/avx_detector.cpp")
if(UNIX)
    target_compile_options(avx_detector PRIVATE -march=native)
elseif(MSVC)
    target_compile_options(avx_detector PRIVATE /EHsc)
endif()
set_property(TARGET avx_detector PROPERTY OUTPUT_NAME avx_detector)
