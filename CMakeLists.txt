cmake_minimum_required(VERSION 3.18)
project(karthik_slm_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CPU Optimization flags for maximum inference throughput
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

add_executable(karthik_server src/cpp/main.cpp)
