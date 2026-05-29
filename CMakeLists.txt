cmake_minimum_required(VERSION 3.16)
project(qsynth LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

add_executable(qsynth src/main.cpp)
target_include_directories(qsynth PRIVATE ${CMAKE_SOURCE_DIR}/include)

enable_testing()
add_executable(qsynth_tests tests/test_qsynth.cpp)
target_include_directories(qsynth_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME qsynth_tests COMMAND qsynth_tests)
