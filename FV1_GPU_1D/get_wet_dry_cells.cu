#include "get_wet_dry_cells.cuh"

__global__ void get_wet_dry_cells
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	AssembledSolution    d_assem_sol,
	int* dry_cells
)
{
	extern __shared__ real hShared[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	if (x < sim_params.cells + 2)
	{
		hShared[tx] = d_assem_sol.h_BC[x];
	}

	__syncthreads();

	real hMax, hBack, hForward, hLocal;

	if (x > 0 && x < sim_params.cells + 1)
	{

		// halo at tx = 0 and tx = blockDim.x - 1 (for blockDim.x = 4, tx = 0, 1, 2, 3)
		hBack = (tx > 0) ? hShared[tx - 1] : d_assem_sol.h_BC[x - 1];
		hLocal = hShared[tx];
		hForward = (tx < blockDim.x - 1) ? hShared[tx + 1] : d_assem_sol.h_BC[x + 1];

		hMax = max(hBack, hLocal);
		hMax = max(hMax, hForward);

		dry_cells[x] = (hMax <= solver_params.tol_dry);
	}
}