#include "init_eta_temp.cuh"

__global__ void init_eta_temp
(
	SimulationParameters sim_params, 
	AssembledSolution    d_assem_sol, 
	real*                etaTemp
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < sim_params.cells + 1) etaTemp[x] = d_assem_sol.h_BC[x] + d_assem_sol.z_BC[x];
}