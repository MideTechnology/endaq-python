cmake_minimum_required(VERSION 3.14)
project(endaq-python)

# for google's unit test framework
set(CMAKE_CXX_STANDARD 11)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(pvss src/pvss.c)

enable_testing()

add_executable(
        pvss_test
        tests/pvss_test.cc
)
target_link_libraries(
        pvss_test
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(pvss_test)
