project(HEATDemoCuda)

######## A simple cmakelists.txt file for Open... #############  
#cmake_minimum_required(VERSION 3.26)  
cmake_minimum_required(VERSION 3.17)
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_C_COMPILER    "/usr/bin/clang-14")
#set(CMAKE_CXX_COMPILER  "/usr/bin/clang++-14")
#set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")


include_directories("../common") 



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

find_package(PNG REQUIRED)
if (PNG_FOUND)
        MESSAGE("{PNG_LIBRARIES}")
        else (PNG_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without PNG")
    endif(PNG_FOUND)


#file(GLOB my_cpp_list "${CMAKE_CURRENT_SOURCE_DIR}/common/*.c")
file(GLOB my_common_list 
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/*.h"
)

if (FALSE)
    message(${CMAKE_CURRENT_SOURCE_DIR})
    foreach(file_path ${my_cpp_list} ${my_common_list})
        message(${file_path})
    endforeach()
endif()


file(GLOB All_SRC
     "*.h"
     "*.cpp"
     "*.cu"
)


#add_executable(HEATDemoCuda main.cpp ${All_SRC} ${my_common_list})

cuda_add_executable(HEATDemoCuda ${All_SRC} ${my_common_list})

include_directories(SYSTEM ${MPI_INCLUDES_PATH})
target_include_directories(HEATDemoCuda PUBLIC ${MPI_CXX_INCLUDE_DIRS} ${PNG_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
#target_link_libraries(HEATDemoCuda PUBLIC ${MPI_CXX_LIBRARIES} ${PNG_LIBRARY} PUBLIC cuda)
#cuda_add_library(HEATDemoCuda PUBLIC ${MPI_CXX_LIBRARIES} ${PNG_LIBRARY} )
#add_library(HEATDemoCuda PUBLIC ${MPI_CXX_LIBRARIES} ${PNG_LIBRARY})

#target_link_libraries(HEATDemoCuda PUBLIC ${MPI_CXX_LIBRARIES} ${PNG_LIBRARY} ${CUDA_LIBRARIES})

target_link_libraries(HEATDemoCuda ${MPI_CXX_LIBRARIES})
target_link_libraries(HEATDemoCuda ${PNG_LIBRARY})
#target_link_libraries(HEATDemoCuda ${CUDA_LIBRARIES})


