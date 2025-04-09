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
    int provided;
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
      MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    }

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
      Kokkos::InitializationSettings settings;
      settings.set_device_id(rank % Kokkos::HIP::detect_device_count());
      if (!Kokkos::is_initialized()) {
        // settings.set_num_threads(2); // if you want ... or more parameters
        Kokkos::initialize(settings);
      }

      {
        int n = 10;
        Kokkos::View<double *, Kokkos::HIP::memory_space> data("data", n);

        Kokkos::parallel_for(
            Kokkos::RangePolicy<Kokkos::HIP::execution_space>(0, n),
            KOKKOS_LAMBDA(const int i) { data(i) = rank * 1.0 + i; });

        Kokkos::fence();

        double local_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::HIP::execution_space>(0, n),
            KOKKOS_LAMBDA(const int i, double &sum) { sum += data(i); },
            local_sum);

        Kokkos::fence();

        double global_sum;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);

        std::cout << "rank[" << rank << "] Lobale Sum : " << local_sum
                  << std::endl;

        if (rank == 0) {
          std::cout << "Globale sum : " << global_sum << std::endl;
        }
      }

      // Kokkos::finalize();

      if (Kokkos::is_initialized()) {
        Kokkos::finalize();
      }
    } catch (std::exception &e) {
      std::cerr << "Exception caught on rank " << rank << ": " << e.what()
                << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Initialized(&initialized);
    if (initialized) {
      MPI_Finalize();
    }
  }
  Kokkos::finalize();
  return 0;
}
