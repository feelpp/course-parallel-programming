cmake_minimum_required(VERSION 3.10)


option(FEELPP_ENABLE_KOKKOS "Enable Kokkos support" ON)
set(Kokkos_ENABLE_OPENMP ON )

if (FEELPP_ENABLE_KOKKOS)
    find_package(Kokkos REQUIRED)
    if ( Kokkos_FOUND )
        message(STATUS "Kokkos found")
        if(Kokkos_ENABLE_OPENMP)
            message(STATUS "Kokkos OpenMP backend is enabled")
        endif()
        if (Kokkos_ENABLE_SERIAL)
            message(STATUS "Kokkos Serial backend is enabled")
        endif()
        if (Kokkos_ENABLE_THREADS)
            message(STATUS "Kokkos Threads backend is enabled")
        endif()
        if(Kokkos_ENABLE_HIP)
            message(STATUS "Kokkos HIP backend is enabled")
        endif()
        if (Kokkos_ENABLE_CUDA)
            message(STATUS "Kokkos CUDA backend is enabled")
        endif()
    endif()
    message(STATUS "[feelpp] Kokkos support enabled")

    # Add an executable
    add_executable(00_views 00_views.cpp)
    add_executable(01_views_1D 01_views_1D.cpp)
    add_executable(02_views_2D 02_views_2D.cpp)
    add_executable(03_views_properties 03_views_properties.cpp)
    add_executable(04_kokkos_exercise_views 04_kokkos_exercise_views.cpp)

    # Link Kokkos to the executable
    target_link_libraries(00_views Kokkos::kokkos)
    target_link_libraries(01_views_1D Kokkos::kokkos)
    target_link_libraries(02_views_2D Kokkos::kokkos)
    target_link_libraries(03_views_properties Kokkos::kokkos)
    target_link_libraries(04_kokkos_exercise_views Kokkos::kokkos)

endif()