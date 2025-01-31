#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>


#include <Kokkos_Core.hpp>


// https://en.wikipedia.org/wiki/Jacobi_polynomials

// Function to evaluate the Jacobi polynomial P_n^(α,β)(x)
KOKKOS_INLINE_FUNCTION
double jacobi_polynomial(int n, double x, double alpha, double beta) {
    if (n == 0) return 1.0;
    if (n == 1) return 0.5 * (alpha + beta + (alpha - beta) * x);

    double p0 = 1.0;
    double p1 = 0.5 * (alpha + beta + (alpha - beta) * x);
    double p_n = 0.0;

    for (int k = 2; k <= n; ++k) {
        p_n = ((2 * k + alpha + beta - 1) * (p1 + (alpha - beta) * p0)) / (k + alpha + beta);
        p_n -= ((k + alpha - 1) * (k + beta - 1) * p0) / ((k - 1) * (k + alpha + beta));
        p0 = p1;
        p1 = p_n;
    }
    return p_n;
}


struct JacobiKernel {
    int n;
    double alpha;
    double beta;
    Kokkos::View<double*> x_values;
    Kokkos::View<double*> results;

    JacobiKernel(int n_, double alpha_, double beta_, Kokkos::View<double*> x_vals, Kokkos::View<double*> res)
        : n(n_), alpha(alpha_), beta(beta_), x_values(x_vals), results(res) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
        results(i) = jacobi_polynomial(n, x_values(i), alpha, beta);
    }
};


int main( int argc, char* argv[] )
{
    Kokkos::initialize(argc, argv);
    {
       const int num_points = 20;
        Kokkos::View<double*> x_values("x_values", num_points);
        Kokkos::View<double*> results("results", num_points);

        // Initialization of input values
        Kokkos::parallel_for("InitX Values", num_points, KOKKOS_LAMBDA(const int i) {
            x_values(i) = -1.0 + 2.0 * i / (num_points - 1); // Values ​​between -1 and 1
        });

        // Evaluation of Jacobi polynomials
        // (degree n=3, α=2.0, β=3.0).
        int n = 3;
        double alpha = 2.0, beta = 3.0; 

        Kokkos::parallel_for("Jacob Eval", num_points, JacobiKernel(n, alpha, beta, x_values, results));

        // Displaying results
        auto h_results = Kokkos::create_mirror_view(results);
        Kokkos::deep_copy(h_results, results);

        for (int i = 0; i < num_points; ++i) {
            std::cout << "P_" << n << "^(" << alpha << ", " << beta << ")(" << h_results(i) << ") = " << h_results(i) << "\n";
        }
        
        // Make a graphic representation afterwards or ...
 
    }
    Kokkos::finalize();
    return 0;
}