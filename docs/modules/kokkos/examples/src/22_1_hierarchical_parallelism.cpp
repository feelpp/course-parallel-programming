#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>


struct HierarchicalParallelism {
  Kokkos::View<double**> matrix;
  HierarchicalParallelism(int N, int M) : matrix("matrix", N, M) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::TeamPolicy<>::member_type& team_member) const {
      const int i = team_member.league_rank();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, matrix.extent(1)),
      [&] (const int j) {
          matrix(i, j) = i * matrix.extent(1) + j;
      });

      team_member.team_barrier();
      if (team_member.team_rank() == 0) {
        double sum = 0.0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team_member, matrix.extent(1)),
            [&] (const int j, double& lsum) {
            lsum += matrix(i, j);
        }, sum);

        Kokkos::single(Kokkos::PerTeam(team_member), [&] () {
            matrix(i, 0) = sum;
            // std::cout << "Sum of row " << i << " is " << sum << std::endl;
        });
      }
  }
};


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
      const int N = 1000;
      const int M = 100;
      HierarchicalParallelism functor(N, M);
      Kokkos::parallel_for(Kokkos::TeamPolicy<>(N, Kokkos::AUTO), functor);
  }
  Kokkos::finalize();
  return 0;
}


// int main(int argc, char *argv[]) {
//   Kokkos::initialize(argc, argv);
//   {
//     Kokkos::Timer timer;
//     const int N = 1000000;
//     const int TEAM_SIZE = 16;
//     const int VECTOR_SIZE = 4;

//     Kokkos::View<double *> data("Data", N);

//     // Data initialization
//     Kokkos::parallel_for(
//         "Init", N, KOKKOS_LAMBDA(const int i) { data(i) = i * 0.01; });

//     double sum = 0.0;

//     // Hierarchical parallelism
//     Kokkos::parallel_reduce(
//         "HierarchicalSum",
//         Kokkos::TeamPolicy<>(N / (TEAM_SIZE * VECTOR_SIZE), TEAM_SIZE,
//                              VECTOR_SIZE),
//         KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team_member,
//                       double &team_sum) {
//           const int team_rank = team_member.team_rank();
//           const int team_size = team_member.team_size();
//           const int league_rank = team_member.league_rank();

//           double thread_sum = 0.0;

//           Kokkos::parallel_reduce(
//               Kokkos::ThreadVectorRange(team_member, VECTOR_SIZE),
//               [&](const int vector_rank, double &vector_sum) {
//                 const int i =
//                     (league_rank * team_size + team_rank) * VECTOR_SIZE +
//                     vector_rank;
//                 if (i < N) {
//                   vector_sum += data(i);
//                 }
//               },
//               thread_sum);

//           Kokkos::single(Kokkos::PerThread(team_member),
//                          [&]() { Kokkos::atomic_add(&team_sum, thread_sum); });
//         },
//         sum);

//     std::cout << "Total Sum : " << sum << std::endl;
//     std::cout << "Average : " << sum / N << std::endl;
//     double elapsed_time = timer.seconds();
//     std::cout << "Elapsed time: " << elapsed_time << " seconds" << std::endl;
//   }
//   Kokkos::finalize();
//   return 0;
// }
