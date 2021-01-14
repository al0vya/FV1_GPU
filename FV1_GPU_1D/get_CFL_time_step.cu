#include "get_CFL_time_step.cuh"

__global__ void get_CFL_time_step
(
	SimulationParameters sim_params,
	SolverParameters     solver_params,
	real                 dx,
	AssembledSolution    d_assem_sol,
	real*                d_dtCFLblockLevel
)
{
	extern __shared__ real dtCFL[];

	int tx = threadIdx.x;
	int x = blockIdx.x * blockDim.x + tx;

	dtCFL[tx] = C(1e7);

	__syncthreads();

	// no sync here because each tx is unique and a write, nothing is being read from so no risk of trying to access an uninitialised value

	if (x > 0 && x < sim_params.cells + 1)
	{
		if (d_assem_sol.h_BC[x] >= solver_params.tol_dry)
		{
			real u = d_assem_sol.q_BC[x] / d_assem_sol.h_BC[x];
			dtCFL[tx] = solver_params.CFL * dx / (abs(u) + sqrt(solver_params.g * d_assem_sol.h_BC[x]));
		}
	}

	__syncthreads();

	for (int blockStride = blockDim.x / 2; blockStride > 0; blockStride >>= 1)
	{
		if (tx < blockStride) dtCFL[tx] = min(dtCFL[tx], dtCFL[tx + blockStride]);

		__syncthreads(); // same logic as before, sync before next read
	}

	if (tx == 0) d_dtCFLblockLevel[blockIdx.x] = dtCFL[0];
}