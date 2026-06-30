cmake_minimum_required(VERSION 3.10)
project(SparsePagedLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(llm_experiment
    src/main.cpp
)

# Otimizações
if(MSVC)
    target_compile_options(llm_experiment PRIVATE /O2)
else()
    target_compile_options(llm_experiment PRIVATE -O3 -march=native)
endif()
