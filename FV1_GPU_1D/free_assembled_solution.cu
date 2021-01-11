#include "free_assembled_solution.cuh"

__host__ void free_assembled_solution(AssembledSolution& d_assembled_solution)
{
	free_device(d_assembled_solution.h_BC);
	free_device(d_assembled_solution.q_BC);
	free_device(d_assembled_solution.z_BC);
	free_device(d_assembled_solution.active_indices);
}