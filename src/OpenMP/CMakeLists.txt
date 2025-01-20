######## A simple cmakelists.txt file for Open..() #############  



cmake_minimum_required(VERSION 3.17)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_COMPILER "/usr/local/bin/g++")
#set(CMAKE_C_COMPILER    "/usr/bin/clang-14")
#set(CMAKE_CXX_COMPILER  "/usr/bin/clang++-14")
#set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")



if(FALSE)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

if(FALSE)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        MESSAGE("{OpenMP_CXX_LIBRARIES}")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        else (OpenMP_CXX_FOUND)
        MESSAGE (SEND_ERROR "This application cannot compile without OpenMP")
    endif(OpenMP_CXX_FOUND)
endif()


if(FALSE)
    find_package(OpenMP REQUIRED)
endif()


file(GLOB WFOPenMP_SRC
     "*.hpp"
     "*.cpp"
)

foreach (myfile ${WFOPenMP_SRC})
    get_filename_component(myname ${myfile} NAME_WLE)
    get_filename_component(dirname ${myfile} DIRECTORY)
    message("${myname}.cpp | ${dir_src}")
    add_executable(${myname} "${myname}.cpp")
    
    #target_compile_options(${myname} PRIVATE ${OpenMP_CXX_FLAGS})
    #target_link_libraries(${myname} PRIVATE ${OpenMP_CXX_FLAGS})

    target_link_libraries(${myname} PUBLIC OpenMP::OpenMP_CXX)    
endforeach (file ${WFOPenMP_SRC})





########### end ####################################  