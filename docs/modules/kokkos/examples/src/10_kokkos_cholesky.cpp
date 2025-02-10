#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

void choleskySimple(Kokkos::View<double **> A, int n) {
  for (int k = 0; k < n; ++k) {
    // Calculation of the diagonal element
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
          double sum = 0.0;
          for (int j = 0; j < k; ++j) {
            sum += A(k, j) * A(k, j);
          }
          A(k, k) = std::sqrt(A(k, k) - sum);
        });

    // Calculation of elements under the diagonal
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(k + 1, n), KOKKOS_LAMBDA(const int i) {
          double sum = 0.0;
          for (int j = 0; j < k; ++j) {
            sum += A(i, j) * A(k, j);
          }
          A(i, k) = (A(i, k) - sum) / A(k, k);
        });
  }

  // Zeroing the upper triangular part
  Kokkos::parallel_for(
      Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const int i) {
        for (int j = i + 1; j < n; ++j) {
          A(i, j) = 0.0;
        }
      });
}

void matrix_product(Kokkos::View<double **> A, Kokkos::View<double **> C,
                    int n) {
  Kokkos::parallel_for(
      "MatrixProduct", n, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < n; j++) { // optimisation
          double sum = 0.0;
          for (int k = 0; k <= i && k <= j; k++) {
            sum += A(i, k) * A(j, k);
          }
          C(i, j) = sum;
        }
      });
}

void matrix_product(Kokkos::View<double **> A, Kokkos::View<double **> B,
                    Kokkos::View<double **> C, int n) {
  Kokkos::parallel_for(
      "MatrixProduct", n, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < n; ++j) {
          double sum = 0.0;
          for (int k = 0; k < n; ++k) {
            sum += A(i, k) * B(k, j);
          }
          C(i, j) = sum;
        }
      });
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int n = 4; // Size of the matrix
    // View allocation
    Kokkos::View<double **> A("A", n, n);
    Kokkos::View<double **> A_original("A_original", n, n);
    Kokkos::View<double **> C("C", n, n);

    // Initialization of the positive defined matrix A
    Kokkos::parallel_for(
        "InitMatrix", n, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < n; ++j) {
            if (i == j) {
              A(i, j) = (i + 1) + (j + 1) + n;
            } else {
              A(i, j) = A(j, i) = std::min(i, j) + 1;
            }
            A_original(i, j) = A(i, j);
          }
        });

    // Synchronization to ensure initialization is complete
    Kokkos::fence();

    // Create mirror views for display
    auto h_A = Kokkos::create_mirror_view(A);
    auto h_A_original = Kokkos::create_mirror_view(A_original);
    auto h_C = Kokkos::create_mirror_view(C);

    // Copy data to host
    Kokkos::deep_copy(h_A, A);

    // Display the initial matrix
    printf("Matrix A init:\n");
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        printf("%f ", h_A(i, j));
      }
      printf("\n");
    }

    // Cholesky factorization
    choleskySimple(A, n);

    // Copy data to host after Cholesky
    Kokkos::deep_copy(h_A, A);

    printf("\nCholesky Matrix (L):\n");
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        printf("%f ", h_A(i, j));
      }
      printf("\n");
    }

    // Calculation of the product L * L^T
    matrix_product(A, C, n);

    // Copy data to host
    Kokkos::deep_copy(h_C, C);

    printf("\nProduct L * L^T :\n");
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        printf("%f ", h_C(i, j));
      }
      printf("\n");
    }

    // Verification
    Kokkos::deep_copy(h_A_original, A_original);
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        max_diff = std::max(max_diff, std::abs(h_C(i, j) - h_A_original(i, j)));
      }
    }
    printf("\nMaximum difference between initial A and L*L^T : %e\n", max_diff);
  }
  Kokkos::finalize();
  return 0;
}
