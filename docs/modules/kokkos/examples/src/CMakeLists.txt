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

    # My prog 
    add_executable(10_kokkos_cholesky 10_kokkos_cholesky.cpp)
    add_executable(11_kokkos_memspace 11_kokkos_memspace.cpp)
    #add_executable(12_kokkos_simple_multi_gpus 12_kokkos_simple_multi_gpus.cpp)
    #add_executable(13_kokkos_execution_patterns 13_kokkos_execution_patterns.cpp)
    #add_executable(14_kokkos_execution_patterns_policies 14_kokkos_execution_patterns_policies.cpp)
    #add_executable(15_kokkos_dual_view 15_kokkos_dual_view.cpp)
    #add_executable(16_kokkos_simd 16_kokkos_simd.cpp)
    #add_executable(17_kokkos_polynom_jacobi_1 17_kokkos_polynom_jacobi_1.cpp)

    # feelpp_add_application(kokkos SRCS feelpp_kokkos.cpp)

    # Link Kokkos to the executable
    target_link_libraries(00_views Kokkos::kokkos)
    target_link_libraries(01_views_1D Kokkos::kokkos)
    target_link_libraries(02_views_2D Kokkos::kokkos)
    target_link_libraries(03_views_properties Kokkos::kokkos)
    target_link_libraries(04_kokkos_exercise_views Kokkos::kokkos)

    # My prog 
    target_link_libraries(10_kokkos_cholesky Kokkos::kokkos)
    target_link_libraries(11_kokkos_memspace Kokkos::kokkos)
    #target_link_libraries(12_kokkos_simple_multi_gpus Kokkos::kokkos)
    #target_link_libraries(13_kokkos_execution_patterns Kokkos::kokkos)
    #target_link_libraries(14_kokkos_execution_patterns_policies Kokkos::kokkos)
    #target_link_libraries(15_kokkos_dual_view Kokkos::kokkos)
    #target_link_libraries(16_kokkos_simd Kokkos::kokkos)
    #target_link_libraries(17_kokkos_polynom_jacobi_1 Kokkos::kokkos);


endif()