

######## A simple cmakelists.txt file for OpenCV() #############  
#cmake_minimum_required(VERSION 3.26)  
cmake_minimum_required(VERSION 3.17)


set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_C_COMPILER    "/usr/bin/clang-14")
#set(CMAKE_CXX_COMPILER  "/usr/bin/clang++-14")
#set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")

project( Exp001 )



#set (OpenCV_DIR /home/cmake/opencv/compiled)
#set( OpenCV_DIR /usr/local/share/OpenCV)


set(OpenCV_STATIC ON)
set(OpenCV_CUDA OFF)
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../install")


find_package( OpenCV REQUIRED )

if (OpenCV_FOUND)
    MESSAGE("{OpenCV_LIBRARIES}")
    else (OpenCV_FOUND)
    MESSAGE (SEND_ERROR "This application cannot compile without OpenCV")
endif(OpenCV_FOUND)

include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories( ${OpenCV_LIB_DIR} )

#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
#    target_link_libraries(A PUBLIC OpenMP::OpenMP_CXX ${OpenCV_LIBS})
#endif()

find_package(MPI REQUIRED)
if (MPI_FOUND)
    MESSAGE("{MPI_CXX_LIBRARIES}")
    else (MPI_FOUND)
    MESSAGE (SEND_ERROR "This application cannot compile without MPI")
endif(MPI_FOUND)


add_executable(Exp001 Exp001.cpp )

include_directories(SYSTEM ${MPI_INCLUDES_PATH})
target_include_directories(Exp001 PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries( Exp001 PUBLIC ${MPI_CXX_LIBRARIES} ${OpenCV_LIBS})

########### end ####################################  




