#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Define a 1D view of doubles with 10 elements
        Kokkos::View<double*> view("view", 10);

        // Initialize the view using parallel_for
        Kokkos::parallel_for("InitView", 10, KOKKOS_LAMBDA(const int i) {
            view(i) = i * 1.0;
        });

        // Print the view elements
        Kokkos::parallel_for("PrintView", 10, KOKKOS_LAMBDA(const int i) {
            printf("view(%d) = %f\n", i, view(i));
        });
    }
    Kokkos::finalize();
    return 0;
}