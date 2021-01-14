#include "add_ghost_cells.cuh"

__global__ void add_ghost_cells
(
	BoundaryConditions   bcs,
	SimulationParameters sim_params,
	AssembledSolution    d_assem_sol
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x == 0)
	{
		d_assem_sol.q_BC[x] = bcs.q_imposed_up > 0 ? bcs.q_imposed_up : d_assem_sol.q_BC[x + 1];
		d_assem_sol.h_BC[x] = bcs.h_imposed_up > 0 ? bcs.h_imposed_up : d_assem_sol.h_BC[x + 1];
		d_assem_sol.z_BC[x] = d_assem_sol.z_BC[x + 1];
	}

	if (x == sim_params.cells + 1)
	{
		d_assem_sol.q_BC[x] = bcs.q_imposed_down > 0 ? bcs.q_imposed_down : d_assem_sol.q_BC[x - 1];
		d_assem_sol.h_BC[x] = bcs.h_imposed_down > 0 ? bcs.h_imposed_down : d_assem_sol.h_BC[x - 1];
		d_assem_sol.z_BC[x] = d_assem_sol.z_BC[x - 1];
	}
}