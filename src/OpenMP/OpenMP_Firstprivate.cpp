#include <omp.h>
#include <stdio.h>

// not using iostream here due to output ordering issues

// iostream tends to output each part between <<'s separately to the console, 
// which can lead to random output if multiple threads are doing the same
// thing.

// printf will generally output the whole result string in one go, so results
// of separate printf calls, even from different threads, will remain intact

// Another fix, other than using printf, would be to give each thread its own 
// place to store output temporarily (a stringstream), and then output the whole
// result in one go.

int main() {
	
	omp_set_num_threads(4);

	// Variable that will be firstprivate
	int val = 123456789;

	printf("Value of \"val\" before the OpenMP parallel region: %d.\n", val);

	#pragma omp parallel firstprivate(val)
	{
		printf("Thread %d sees \"val\" = %d, and updates it to be %d.\n", omp_get_thread_num(), val, omp_get_thread_num());
		val = omp_get_thread_num();
	}

	// Value after the parallel region; unchanged.
	printf("Value of \"val\" after the OpenMP parallel region: %d.\n", val);

	return 0;
}

