#include "malloc_assembled_solution.cuh"

__host__ void malloc_assembled_solution
(
	AssembledSolution& d_assembled_solution,
	size_t             count
)
{	
	size_t bytes_real = count * sizeof(real);
	size_t bytes_int  = count * sizeof(int);

	d_assembled_solution.h_BC           = (real*)malloc_device(bytes_real);
	d_assembled_solution.q_BC           = (real*)malloc_device(bytes_real);
	d_assembled_solution.z_BC           = (real*)malloc_device(bytes_real);
	d_assembled_solution.active_indices = (int*) malloc_device(bytes_int);
	d_assembled_solution.length         = count;
}