
project(Test)

cmake_minimum_required(VERSION 3.22)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${example_name} LANGUAGES CXX)

set(GPU_RUNTIME "HIP" CACHE STRING "Switches between HIP and CUDA")
set(GPU_RUNTIMES "HIP" "CUDA")
set_property(CACHE GPU_RUNTIME PROPERTY STRINGS ${GPU_RUNTIMES})

if(NOT "${GPU_RUNTIME}" IN_LIST GPU_RUNTIMES)
    set(ERROR_MESSAGE "GPU_RUNTIME is set to \"${GPU_RUNTIME}\".\nGPU_RUNTIME must be either HIP or CUDA.")
    message(FATAL_ERROR ${ERROR_MESSAGE})
endif()

enable_language(${GPU_RUNTIME})
set(CMAKE_${GPU_RUNTIME}_STANDARD 17)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)


set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")


list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")


file(GLOB All_SRC
     "*.h"
     "*.cpp"
)



add_executable(Test Test.cpp ${All_SRC})

set(include_dirs "../../Common")
# For examples targeting NVIDIA, include the HIP header directory.
if(GPU_RUNTIME STREQUAL "CUDA")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
endif()

#target_include_directories(${example_name} PRIVATE ${include_dirs})
set_source_files_properties(Test.cpp PROPERTIES LANGUAGE ${GPU_RUNTIME})

