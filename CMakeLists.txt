cmake_minimum_required(VERSION 3.18)
project(SparsePagedLLM CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architecture Flags for Maximum Performance
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# Sources and headers
set(CPU_SOURCES src/main.cpp)
set(CUDA_SOURCES src/cuda_kernels.cu)

# CPU-only executable (fallback)
add_executable(llm_experiment ${CPU_SOURCES})
if(MSVC)
    target_compile_options(llm_experiment PRIVATE /O2)
else()
    target_compile_options(llm_experiment PRIVATE -O3 -march=native)
endif()

# CUDA-enabled executable
add_executable(llm_cuda ${CUDA_SOURCES} ${CPU_SOURCES})
target_compile_definitions(llm_cuda PRIVATE USE_CUDA)

if(MSVC)
    target_compile_options(llm_cuda PRIVATE /O2)
    # Pass NVCC options for compiler version bypass
    target_compile_options(llm_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)
else()
    target_compile_options(llm_cuda PRIVATE -O3)
endif()

# Set target compile options for CUDA
target_compile_options(llm_cuda PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-use_fast_math;-ftz=true;-prec-div=false;-prec-sqrt=false>
)
# Note: CMake automatically links cudart when project(..., CUDA) is declared.
