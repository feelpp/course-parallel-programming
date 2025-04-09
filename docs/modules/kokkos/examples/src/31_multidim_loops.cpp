#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int N1 = 10;
        const int N2 = 10;
        const int N3 = 10;

        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {N1, N2, N3});

        Kokkos::parallel_for("3DLoop", policy, KOKKOS_LAMBDA(const int i, const int j, const int k) {
            // Example computation
            printf("Processing element (%d, %d, %d)\n", i, j, k);
        });
    }
    Kokkos::finalize();
    return 0;
}
