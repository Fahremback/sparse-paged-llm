cmake_minimum_required(VERSION 3.10)
project(SparsePagedLLM CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

# CUDA Architecture Flags for Maximum Performance
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-ftz=true;-prec-div=false;-prec-sqrt=false")

# Source files
set(CPU_SOURCES src/main.cpp)
set(CUDA_SOURCES src/cuda_kernels.cu)
set(HEADERS src/types.h src/router.h src/pager.h src/cuda_kernels.cu src/cuda_pager.h)

# CPU-only executable (fallback)
add_executable(llm_experiment ${CPU_SOURCES} ${HEADERS})
if(MSVC)
    target_compile_options(llm_experiment PRIVATE /O2)
else()
    target_compile_options(llm_experiment PRIVATE -O3 -march=native)
endif()

# CUDA-enabled executable
cuda_add_executable(llm_cuda ${CUDA_SOURCES} ${HEADERS})
target_compile_definitions(llm_cuda PRIVATE USE_CUDA)
if(MSVC)
    target_compile_options(llm_cuda PRIVATE /O2)
else()
    target_compile_options(llm_cuda PRIVATE -O3)
endif()
target_link_libraries(llm_cuda ${CUDA_LIBRARIES} cudart)
