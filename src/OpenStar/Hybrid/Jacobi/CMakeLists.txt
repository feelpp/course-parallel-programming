
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

if(TRUE)
    find_package(OpenMP REQUIRED)
endif()




file(GLOB WFOPenMP_SRC
     "*.h"
     "*.c"
)

foreach (myfile ${WFOPenMP_SRC})
    get_filename_component(myname ${myfile} NAME_WLE)
    get_filename_component(dirname ${myfile} DIRECTORY)
    message("${myname}.c | ${dir_src}")
    add_executable(${myname} "${myname}.c")

    #TARGET_LINK_LIBRARIES(${myname} -lm ${LIBS})

    include_directories(SYSTEM ${MPI_INCLUDES_PATH})
    target_include_directories(${myname}  PUBLIC ${MPI_CXX_INCLUDE_DIRS} PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(${myname} PUBLIC ${MPI_CXX_LIBRARIES} PUBLIC OpenMP::OpenMP_CXX)
    
    #if(OpenMP_CXX_FOUND)
    #    target_link_libraries(${myname} PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS})
    #endif()
endforeach (file ${WFOPenMP_SRC})

########### end ####################################  
