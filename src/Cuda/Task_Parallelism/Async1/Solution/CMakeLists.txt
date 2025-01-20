
#project(MPI_bcast)

######## A simple cmakelists.txt file for ... #############  

cmake_minimum_required(VERSION 3.17)
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_C_COMPILER    "/usr/bin/clang-14")
#set(CMAKE_CXX_COMPILER  "/usr/bin/clang++-14")
#set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")



if(FALSE)
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
        MESSAGE (SEND_ERROR "This application cannot compile without OpenMPI")
    endif(OpenMP_CXX_FOUND)
endif()


find_package(CUDA REQUIRED)

if (CUDA_FOUND)
        MESSAGE("{CUDA_CXX_LIBRARIES}")
        else (CUDA_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without CUDA")
    endif(CUDA_FOUND)

add_definitions(-D_FORCE_INLINES)

#set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_21 -std=c++11)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 )

file(GLOB WFOPenMP_SRC
     "*.cu"
     "*.h"
)

foreach (myfile ${WFOPenMP_SRC})
    get_filename_component(myname ${myfile} NAME_WLE)
    get_filename_component(dirname ${myfile} DIRECTORY)
    message("${myname}.cu | ${dir_src}")
    #add_executable(${myname} "${myname}.c")

    cuda_add_executable(${myname} "${myname}.cu")

    #target_link_libraries( ${myname} -lfoobar -ljoestuff )

    #if(MPI_FOUND)
        #include_directories(SYSTEM ${MPI_INCLUDES_PATH})
        #target_include_directories(${myname}  PUBLIC ${MPI_CXX_INCLUDE_DIRS})
        #target_link_libraries(${myname}  PUBLIC ${MPI_CXX_LIBRARIES} )
    #endif()


endforeach (file ${WFOPenMP_SRC})



file(GLOB WFOPenMP_SRC
     "*.cpp"
     "*.h"
)

foreach (myfile ${WFOPenMP_SRC})
    get_filename_component(myname ${myfile} NAME_WLE)
    get_filename_component(dirname ${myfile} DIRECTORY)
    message("${myname}.cpp | ${dir_src}")
    #add_executable(${myname} "${myname}.c")

    cuda_add_executable(${myname} "${myname}.cpp")

    #target_link_libraries( ${myname} -lfoobar -ljoestuff )

    #if(MPI_FOUND)
        #include_directories(SYSTEM ${MPI_INCLUDES_PATH})
        #target_include_directories(${myname}  PUBLIC ${MPI_CXX_INCLUDE_DIRS})
        #target_link_libraries(${myname}  PUBLIC ${MPI_CXX_LIBRARIES} )
    #endif()
endforeach (file ${WFOPenMP_SRC})



########### end ####################################  
