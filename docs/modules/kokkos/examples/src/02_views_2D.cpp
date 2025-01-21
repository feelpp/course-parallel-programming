#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Define a 2D view of doubles with 10 elements
        Kokkos::View<double**> view("view", 10, 2);

        // Initialize the view using parallel_for
        Kokkos::parallel_for("InitView", 10, KOKKOS_LAMBDA(const int i) {
            view(i, 0) = i * 1.0;
            view(i, 1) = i * 2.0;
        });

        // Print the view elements
        Kokkos::parallel_for("PrintView", 10, KOKKOS_LAMBDA(const int i) {
            printf("view(%d) = %f\n", i, view(i, 0));
            printf("view(%d) = %f\n", i, view(i, 1));
        });
    }
    Kokkos::finalize();
    return 0;
}