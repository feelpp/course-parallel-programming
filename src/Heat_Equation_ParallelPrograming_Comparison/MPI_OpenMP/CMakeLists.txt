project(HEATDemoHybrid)

######## A simple cmakelists.txt file for Open... #############  
#cmake_minimum_required(VERSION 3.26)  
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


include_directories("../common") 



if(TRUE)
    find_package(MPI REQUIRED)
    if (MPI_FOUND)
        MESSAGE("{MPI_CXX_LIBRARIES}")
        else (MPI_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without MPI")
    endif(MPI_FOUND)
endif()

if(TRUE)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        MESSAGE("{OpenMP_CXX_LIBRARIES}")
        else (OpenMP_CXX_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without OpenMP")
    endif(OpenMP_CXX_FOUND)
endif()


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

message(${CMAKE_CURRENT_SOURCE_DIR})
foreach(file_path ${my_cpp_list} ${my_common_list})
    message(${file_path})
endforeach()


file(GLOB All_SRC
     "*.h"
     "*.cpp"
     "*.c"
)


add_executable(HEATDemoHybid main.cpp ${All_SRC} ${my_common_list})


include_directories(SYSTEM ${MPI_INCLUDES_PATH})
target_include_directories(HEATDemoHybid PUBLIC ${MPI_CXX_INCLUDE_DIRS} PUBLIC OpenMP::OpenMP_CXX ${PNG_INCLUDE_DIR})
target_link_libraries(HEATDemoHybid PUBLIC ${MPI_CXX_LIBRARIES} PUBLIC OpenMP::OpenMP_CXX ${PNG_LIBRARY})
