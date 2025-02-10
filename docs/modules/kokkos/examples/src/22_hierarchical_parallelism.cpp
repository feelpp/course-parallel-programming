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
    Kokkos::Timer timer;
    const int N = 1000000;
    const int TEAM_SIZE = 16;
    const int VECTOR_SIZE = 4;

    Kokkos::View<double *> data("Data", N);

    // Data initialization
    Kokkos::parallel_for(
        "Init", N, KOKKOS_LAMBDA(const int i) { data(i) = i * 0.01; });

    double sum = 0.0;

    // Hierarchical parallelism
    Kokkos::parallel_reduce(
        "HierarchicalSum",
        Kokkos::TeamPolicy<>(N / (TEAM_SIZE * VECTOR_SIZE), TEAM_SIZE,
                             VECTOR_SIZE),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member,
                      double &team_sum) {
          const int team_rank = team_member.team_rank();
          const int team_size = team_member.team_size();
          const int league_rank = team_member.league_rank();

          double thread_sum = 0.0;

          Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team_member, VECTOR_SIZE),
              [&](const int vector_rank, double &vector_sum) {
                const int i =
                    (league_rank * team_size + team_rank) * VECTOR_SIZE +
                    vector_rank;
                if (i < N) {
                  vector_sum += data(i);
                }
              },
              thread_sum);

          Kokkos::single(Kokkos::PerThread(team_member),
                         [&]() { Kokkos::atomic_add(&team_sum, thread_sum); });
        },
        sum);

    std::cout << "Total Sum : " << sum << std::endl;
    std::cout << "Average : " << sum / N << std::endl;
    double elapsed_time = timer.seconds();
    std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
