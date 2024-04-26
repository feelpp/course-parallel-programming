

project(main)

cmake_minimum_required(VERSION 3.22)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)



#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD 23)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MD /02")
#string(REGEX REPLACE "/MD" "/02" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


if(TRUE)
    find_package(MPI REQUIRED)
    if (MPI_FOUND)
        MESSAGE("{MPI_CXX_LIBRARIES}")
        else (MPI_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without MPI")
    endif(MPI_FOUND)
endif()

if(FALSE)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        MESSAGE("{OpenMP_CXX_LIBRARIES}")
        else (OpenMP_CXX_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without OpenMP")
    endif(OpenMP_CXX_FOUND)
endif()

if(FALSE)
    find_package(CUDA REQUIRED)
    if (CUDA_FOUND)
            MESSAGE("{CUDA_CXX_LIBRARIES}")
            MESSAGE(STATUS "Found headers CUDA : ${CUDA_INCLUDE_DIRS}")
            MESSAGE(STATUS "Found lib CUDA : ${CUDA_LIBRARIES}")
            MESSAGE(STATUS "Found CUDA nvcc : ${CUDA_NVCC_EXECUTABLE}")
            else (CUDA_FOUND)
            MESSAGE (SEND_ERROR "This application cannot compile without CUDA")
        endif(CUDA_FOUND)

    add_definitions(-D_FORCE_INLINES)
    #set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21 -std=c++11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 )
endif()

#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)


#find_package(TBB REQUIRED)
#if (TBB_FOUND)
#    MESSAGE("{TBB_CXX_LIBRARIES}")
#endif(TBB_FOUND)


MESSAGE("{COMPILE_WITH_HIP}")
#include_directories(/opt/rocm/hip/include)
include_directories(/opt/rocm-5.7.2/hip/include)
#list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.7.2 /opt/rocm-5.7.2)
# Find hip
#find_package(hip)
# Set the GPU to compile for
#set(GPU_TARGETS "gfx900:gfx906:gfx1032;gfx1035" CACHE STRING "GPU targets to compile for")

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

#set(ROCM_ROOT "/opt/rocm" CACHE PATH "Root directory of the ROCm installation")
set(ROCM_ROOT "/opt/rocm-5.7.2" CACHE PATH "Root directory of the ROCm installation")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")


message(CURRENT DIR= [${CMAKE_CURRENT_SOURCE_DIR}])

#BEGIN::SPECX

file(GLOB_RECURSE Specx_list 
    #"${CMAKE_CURRENT_SOURCE_DIR}/../../specxpp/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx/*.hpp"
)

if (TRUE)
    foreach(file_path ${Specx_list})
        message(${file_path})
    endforeach()
endif()


MACRO(HEADER_DIRECTORIES return_list)
    FILE(GLOB_RECURSE new_list ${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx/Src/*.hpp)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

HEADER_DIRECTORIES(header_specx_dir_list)

list(LENGTH header_specx_dir_list header_dir_list_count)
message(STATUS "[INFO] Found ${header_dir_list_count} header directories.")

if (TRUE)
    foreach(dir_path ${header_specx_dir_list})
        message(${dir_path})
    endforeach()
endif()

#include_directories("/home/u2/lemoine/SpecxProjects/specxpp/build/Src") 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx/build/Src) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx/Src) 

#END::SPECX


include_directories(SYSTEM ${MPI_INCLUDES_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Test13)
set(include_dirs "../Test13")

file(GLOB SOURCES
    "*.cpp"
)

add_executable(main ${SOURCES} ${Specx_list})

#/home/u2/lemoine/Specx20/specx/build/Src/libspecx.a

target_link_libraries(main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../Specx20/specx/build/Src/libspecx.a 
    PUBLIC 
        ${MPI_CXX_LIBRARIES}
)

#target_include_directories(MiniTest PRIVATE "/home/u2/lemoine/SpecxProjects/specxpp")

#if(GPU_RUNTIME STREQUAL "CUDA")
#    list(APPEND include_dirs "${ROCM_ROOT}/include")
#endif()

if(GPU_RUNTIME STREQUAL "HIP")
    list(APPEND include_dirs "${ROCM_ROOT}/include")
    MESSAGE(${ROCM_ROOT}/include)
endif()

set_source_files_properties(main.cpp PROPERTIES LANGUAGE ${GPU_RUNTIME})

target_include_directories(
    main
    PRIVATE
        ${header_specx_dir_list}
    PUBLIC 
        ${MPI_CXX_INCLUDE_DIRS}
)

